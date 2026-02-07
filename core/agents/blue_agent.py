# core/agents/blue_agent.py — ARIASKA BlueAgent v11.1 Sentinel Prime
# 🛡️ Defensive AI | Anomaly Detection | Intrusion Response | GPT-Enhanced Threat Defense

import os
import random
import subprocess
import time
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
from core.utils.stats_monitor import StatsMonitor
from core.environment.cyber_environment import CyberEnvironment
from core.teach.teach import TeachModule
from core.ui_helpers import display_status_bar
from core.ui_helpers import get_action_description
from core.interfaces.memory_sync_interface import MemorySyncInterface
from core.utils.replay_buffer import ReplayBuffer
from core.utils.gpt_cache_handler import GPTCacheHandler
from core.interfaces.agent_interface import AgentInterface
from core.gpt_manager import GPTManager
from core.multiagent.memory_router import MemoryRouter

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
    # Add property definitions for agent_id and role
    @property
    def agent_id(self):
        return self._agent_id
        
    @agent_id.setter
    def agent_id(self, value):
        self._agent_id = value
        
    @property
    def role(self):
        return self._role
        
    @role.setter
    def role(self, value):
        self._role = value
    
    def __init__(
        self,
        agent_id="BlueAgent",
        role="CyberDefense",
        device="cuda",
        agent_manager=None,
        memory_router=None,
        memory_manager=None,
        verbosity="standard",
        gpt_manager=None,  # PHASE 0 FIX: Accept injected GPTManager
    ):
        self._agent_id = agent_id  # Use internal attribute for property
        self._role = role  # Use internal attribute for property
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        device_str = str(self.device)  # Convert device to string for network constructors
        self.policy_net = PolicyNet(input_size=512, output_size=5, device=device_str).to(self.device)
        self.value_net = ValueNet(input_size=512, device=device_str).to(self.device)
        self.target_value_net = ValueNet(input_size=512, device=device_str).to(self.device)
        self.target_value_net.load_state_dict(self.value_net.state_dict())
        self.target_value_net.eval()
        self.target_net = PolicyNet(input_size=512, output_size=5, device=device_str).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Initialize optimizers and schedulers
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=1e-4)
        self.policy_scheduler = torch.optim.lr_scheduler.StepLR(self.policy_optimizer, step_size=1000, gamma=0.95)
        self.value_scheduler = torch.optim.lr_scheduler.StepLR(self.value_optimizer, step_size=1000, gamma=0.95)
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
        # PHASE 0 FIX: Use injected GPTManager or create one if not provided
        # AgentManager will override this with the shared instance via _sync_gpt_context
        self.gpt_manager = gpt_manager if gpt_manager is not None else GPTManager()
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
        # All LLM functionality now handled by self.gpt_manager

    def _init_multiagent_links(self):
        if self.agent_manager:
            self.red = self.agent_manager.get_agent("RedAgent")
            self.orion = self.agent_manager.get_agent("OrionAgent")
        else:
            self.red = None
            self.orion = None
        # All LLM functionality now handled by self.gpt_manager

    def react_to_action(self, red_action: str) -> dict:
        """
        PHASE 0 FIX: React to RedAgent's action with defensive measures.
        
        This method analyzes the Red action and returns a structured defense dict
        that will be processed by env._process_blue_defense() WITHOUT calling env.step().
        
        Args:
            red_action: The command executed by RedAgent
            
        Returns:
            dict with keys:
                - honeypots_deployed: list of honeypot names to deploy
                - credentials_reset: bool, whether to reset credentials
                - alert_increase: float, amount to increase alert level
        """
        defense_result = {
            "honeypots_deployed": [],
            "credentials_reset": False,
            "alert_increase": 0.0
        }
        
        try:
            # Get current state from shared environment
            state = self.env.get_global_state() if hasattr(self, "env") and self.env else {}
            alert_level = state.get("blue_team_alert", 0)
            detection_risk = state.get("detection_risk", 0)
            
            # Analyze Red's action for threat level
            red_action_lower = red_action.lower() if red_action else ""
            
            # High-risk offensive actions trigger immediate response
            if any(kw in red_action_lower for kw in ["exploit", "metasploit", "msfconsole", "payload"]):
                defense_result["alert_increase"] = 15.0
                defense_result["honeypots_deployed"].append("exploit_honeypot")
                if self.verbosity in ("debug", "verbose"):
                    console.print(f"[blue]🛡️ BlueAgent: Detected exploit attempt, deploying honeypot[/blue]")
                    
            elif any(kw in red_action_lower for kw in ["privesc", "sudo", "su ", "linpeas", "winpeas"]):
                defense_result["alert_increase"] = 20.0
                defense_result["credentials_reset"] = True  # Reset to block privesc
                if self.verbosity in ("debug", "verbose"):
                    console.print(f"[blue]🔐 BlueAgent: Privesc detected, resetting credentials[/blue]")
                    
            elif any(kw in red_action_lower for kw in ["nmap", "masscan", "rustscan"]):
                defense_result["alert_increase"] = 5.0
                if alert_level > 50:
                    defense_result["honeypots_deployed"].append("scan_decoy")
                    
            elif any(kw in red_action_lower for kw in ["gobuster", "dirsearch", "ffuf", "feroxbuster"]):
                defense_result["alert_increase"] = 8.0
                defense_result["honeypots_deployed"].append("web_honeypot")
                
            elif any(kw in red_action_lower for kw in ["scp", "exfil", "wget", "curl", "nc "]):
                defense_result["alert_increase"] = 25.0
                defense_result["credentials_reset"] = True
                if self.verbosity in ("debug", "verbose"):
                    console.print(f"[red]🚨 BlueAgent: Exfiltration attempt detected![/red]")
            
            # High alert mode triggers additional defenses
            if alert_level > 70:
                self.current_mode = "Defensive"
                if not defense_result["honeypots_deployed"]:
                    defense_result["honeypots_deployed"].append("high_alert_honeypot")
                    
            # GPT-enhanced defense for complex situations
            if alert_level > 60 or detection_risk > 5.0:
                if self.gpt_calls_this_episode < self.gpt_call_limit:
                    try:
                        defense_prompt = (
                            f"RedAgent action: {red_action}. Alert: {alert_level}. Risk: {detection_risk}. "
                            "Should we deploy honeypot, reset credentials, or both? Reply: honeypot/reset/both/none"
                        )
                        gpt_response = self.query_tactical_gpt(defense_prompt)
                        gpt_response_lower = gpt_response.lower() if gpt_response else ""
                        
                        if "honeypot" in gpt_response_lower or "both" in gpt_response_lower:
                            defense_result["honeypots_deployed"].append("gpt_suggested_honeypot")
                        if "reset" in gpt_response_lower or "both" in gpt_response_lower:
                            defense_result["credentials_reset"] = True
                            
                        self.gpt_calls_this_episode += 1
                    except Exception as e:
                        if self.verbosity in ("debug", "verbose"):
                            console.print(f"[yellow]⚠ GPT defense query failed: {e}[/yellow]")
                            
        except Exception as e:
            console.print(f"[yellow]⚠ BlueAgent react_to_action error: {e}[/yellow]")
            
        return defense_result

    def query_tactical_gpt(self, prompt, complexity="standard"):
        """
        Use GPT-4o-mini for tactical advice with appropriate complexity handling.
        """
        try:
            # Enhanced prompt for tactical decisions
            tactical_prompt = (
                "As a cybersecurity blue team defense specialist, provide a concise tactical recommendation. "
                "Focus on defensive strategies and countermeasures. "
                "Avoid verbose explanations. Respond with actionable advice only.\n\n"
                f"Task: {prompt.strip()}"
            )
            
            # Use GPTManager with appropriate task type
            if complexity == "complex":
                return self.gpt_manager.smart_decision(
                    task_type="complex_defense", 
                    task_description=tactical_prompt
                )
            else:
                return self.gpt_manager.gpt_request(
                    tactical_prompt, 
                    task_type="tactical", 
                    agent_id=self.agent_id
                )
                
        except Exception as e:
            console.print(f"[red]❌ query_tactical_gpt error: {e}[/red]")
            return self.gpt_manager.smart_decision(task_type="defense", task_description=prompt)

    def _postprocess_gpt_output(self, output: str) -> str:
        """
        Remove verbose/AI disclaimers from GPT output.
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
            # Use the output_size from the policy_net properly
            action_idx = random.randint(0, getattr(self.policy_net, 'output_size', 5) - 1)
            if self.verbosity in ("debug", "verbose"):
                console.print(f"[yellow]🎲 Random action selected: {get_action_description(action_idx)}[/yellow]")
            return action_idx
        phase_vec = get_phase_vector(phase, str(self.device))
        q_values = self.policy_net(state_tensor.unsqueeze(0), phase_vector=phase_vec)
        action_idx = torch.argmax(q_values, dim=-1).item()
        if self.verbosity in ("debug", "verbose"):
            console.print(f"[cyan]🎯 PolicyNet action selected: {get_action_description(action_idx)}[/cyan]")
        # PHASE 2A FIX: Removed GPT mode-switch call from select_action.
        # Mode switching is now handled by alert_level checks in simulate_step.
        return action_idx
    
    def act(self, state: dict) -> dict:
        """Act method for Phase 2 compatibility."""
        try:
            # Blue agent focuses on defense and monitoring
            detection_level = state.get('blue_team_alert', 0)
            detection_risk = state.get('detection_risk', 0)
            
            # Defensive actions based on threat level
            if detection_level > 70:
                action = "sudo ufw enable && sudo fail2ban-client status"
                response_type = "high_alert_response"
                reward = 15.0  # High reward for proper response
            elif detection_level > 40:
                action = "netstat -tulpn | grep LISTEN"
                response_type = "monitoring_response"
                reward = 8.0
            elif detection_risk > 50:
                action = "sudo tail -f /var/log/auth.log | grep FAILED"
                response_type = "threat_investigation"
                reward = 10.0
            else:
                action = "ps aux | grep -E '(ssh|nc|nmap)'"
                response_type = "routine_monitoring"
                reward = 5.0
            
            # Blue agent always succeeds in defensive actions
            success = True
            
            return {
                'action': action,
                'success': success,
                'reward': reward,
                'info': {
                    'response_type': response_type,
                    'detection_level': detection_level,
                    'agent_role': 'defense'
                }
            }
        except Exception as e:
            # Fallback defensive action
            return {
                'action': 'ps aux',
                'success': True,
                'reward': 2.0,
                'info': {'error': str(e), 'fallback': True}
            }

    def simulate_step(self, episode=1, step=1, shared_context=None):
        """
        PHASE 0 REFACTOR: BlueAgent simulate_step now does NOT call env.step().
        
        Instead, Blue monitors the environment state and decides on defensive actions.
        The actual defense is applied via react_to_action() called by AgentManager.
        Blue's reward is computed based on detection success, not env.step() return.
        """
        try:
            # Get current state from shared environment (don't reset - Red controls that)
            if step == 1:
                # On first step, just get state without resetting
                state = self.env.get_global_state() if hasattr(self, "env") and self.env else {}
            else:
                state = getattr(self, "_last_state", self.env.get_global_state() if self.env else {})
                
            # Inject ScoutAgent_phase if available
            # PHASE 2A FIX: Use shared_context phase, don't call scout.advise_phase()
            if shared_context and "ScoutAgent_phase" in shared_context:
                state["phase"] = shared_context["ScoutAgent_phase"]
            elif shared_context and "phase" in shared_context:
                state["phase"] = shared_context["phase"]
            else:
                state["phase"] = state.get("phase", "recon")
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
                action = "defensive_monitoring"
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
            
            # PHASE 0 FIX: Do NOT call env.step() here!
            # Blue's reward is based on detection success, not environment transition
            # The AgentManager will call react_to_action() to apply Blue's defense
            
            # Calculate Blue's reward based on defensive metrics
            alert_level = state.get("blue_team_alert", 0)
            detection_risk = state.get("detection_risk", 0)
            
            # Blue reward: positive for high detection, negative for Red success
            if alert_level > 70:
                reward = 10.0  # High alert = Blue detected Red
            elif alert_level > 40:
                reward = 5.0   # Moderate detection
            elif detection_risk > 5.0:
                reward = -5.0  # Red is being stealthy, bad for Blue
            else:
                reward = 2.0   # Baseline monitoring reward
                
            # next_state is just current state (no env transition)
            next_state = state
            
            self.command_history.append(str(action))
            self.stats_monitor.log_step(self.agent_id, str(action), reward=reward)
            self._last_state = next_state
            self.last_output = f"Action: {action}, Reward: {reward}, Phase: {state.get('phase', 'N/A')}"
            reasoning_key = f"{action}|{state.get('phase')}"
            if reasoning_key in self.gpt_reasoning_cache:
                self.last_reasoning = self.gpt_reasoning_cache[reasoning_key]
            else:
                # PHASE 2A FIX: Only call GPT for reasoning if under call limit
                if self.gpt_calls_this_episode < self.gpt_call_limit:
                    self.last_reasoning = self.gpt_handler.query(
                        f"Explain why action {action} is optimal for phase {state.get('phase')}.",
                        model="gpt-5.1-codex-mini",
                    )
                    self.gpt_reasoning_cache[reasoning_key] = self.last_reasoning
                    self.gpt_calls_this_episode += 1
                else:
                    self.last_reasoning = f"Action {action} selected for phase {state.get('phase', 'N/A')} (GPT budget reached)"
            display_status_bar(self.agent_id, episode, step)
            self._log_training_step(episode, step, str(action), state, self.last_output)
            if self.verbosity in ("debug", "verbose"):
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
                if self.verbosity in ("debug", "verbose"):
                    console.print(
                        "[red]🚨 BlueAgent: Alert level high, activating countermeasures![/red]"
                    )
                if hasattr(self.env, "honeypots"):
                    self.env.honeypots.append("fake_ssh")
            
            # PHASE 2A FIX: Consolidated GPT defense — ONE call max for defense
            # Only query GPT once for defense when needed, instead of 2-3 separate calls
            red_agent = getattr(self.agent_manager, "red_agent", None) if self.agent_manager else None
            red_command = None
            needs_gpt_defense = False
            
            if red_agent and hasattr(red_agent, "command_history") and red_agent.command_history:
                red_command = red_agent.command_history[-1]
                if "exploit" in str(red_command) or "privesc" in str(red_command):
                    needs_gpt_defense = True
                    
            if state.get("blue_team_alert", 0) > 60 or state.get("detection_risk", 0) > 5.0:
                needs_gpt_defense = True
            
            if needs_gpt_defense and self.gpt_calls_this_episode < self.gpt_call_limit:
                try:
                    consolidated_prompt = (
                        f"Alert: {alert_level}, Risk: {detection_risk}. "
                    )
                    if red_command:
                        consolidated_prompt += f"RedAgent action: {red_command}. "
                    consolidated_prompt += (
                        "Provide one-line defensive recommendation: honeypot/reset/lockdown/monitor."
                    )
                    gpt_defense = self.query_tactical_gpt(consolidated_prompt)
                    self.gpt_calls_this_episode += 1
                    gpt_defense_lower = gpt_defense.lower() if gpt_defense else ""
                    
                    if "honeypot" in gpt_defense_lower and hasattr(self.env, "honeypots"):
                        self.env.honeypots.append("gpt_defense_honeypot")
                    if "lockdown" in gpt_defense_lower:
                        self.current_mode = "Defensive"
                    if self.verbosity in ("debug", "verbose"):
                        console.print(f"[cyan]🛡️ GPT Defense: {gpt_defense}[/cyan]")
                except Exception:
                    pass
            
            if self.verbosity != "silent":
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
            # Log experience to MemoryRouter for global observability
            if self.memory_router and hasattr(self.memory_router, 'log_transition'):
                self.memory_router.log_transition(
                    self.agent_id,
                    state,
                    action,
                    reward,
                    next_state,
                    priority=abs(reward)+0.01,
                    gpt_tokens=gpt_tokens
                )
            # --- Target network update every target_update_freq steps ---
            if self.step_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            # Log epsilon for monitoring
            if (step == 1 or step % 10 == 0) and self.verbosity in ("debug", "verbose"):
                console.print(f"[cyan]Epsilon: {self.epsilon:.4f}[/cyan]")
            # LLM usage summary (stub, see below)
            if step == 1 or step % 10 == 0:
                gpt_calls = self.gpt_manager.get_token_usage()
                console.print(f"[blue]LLM usage: GPT calls: {gpt_calls}[/blue]")
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
                stats = self.stats_monitor._get_agent_stats(self.agent_id)
                last_reward = stats.rewards[-1] if stats.rewards else 0
                self.last_rewards.append(last_reward)
                if len(self.last_rewards) > no_progress_steps:
                    self.last_rewards.pop(0)
                if (
                    len(self.last_rewards) == no_progress_steps
                    and max(self.last_rewards) - min(self.last_rewards) < 1e-3
                ):
                    stuck = True
            else:
                stats = self.stats_monitor._get_agent_stats(self.agent_id)
                last_reward = stats.rewards[-1] if stats.rewards else 0
                self.last_rewards = [last_reward]
            if stuck:
                console.print(f"[yellow]⚠ {self.agent_id} appears stuck. Triggering GPT-based recovery.[/yellow]")
                recovery_prompt = (
                    f"Agent {self.agent_id} has repeated action '{self.last_action}' {stuck_window} times with no success. "
                    f"Recent rewards: {getattr(self, 'last_rewards', [])}. "
                    "Suggest an alternative defensive strategy or command to escape this local minimum. Respond ONLY with the command."
                )
                recovery_cmd = self.gpt_manager.gpt_request(
                    recovery_prompt, task_type="reasoning", agent_id=self.agent_id
                )
                if recovery_cmd and isinstance(recovery_cmd, str) and len(recovery_cmd.split()) > 1:
                    action = recovery_cmd
                    console.print(f"[green]🧠 GPT Recovery Command Applied: {action}[/green]")
                    self.repetition_count[action] = 0
            return {
                "command": str(action),
                "phase": state.get("phase", "N/A"),
                "reward": float(reward),
                "gpt_calls": self.stats_monitor._get_agent_stats(self.agent_id).gpt_calls,
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
        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.policy_optimizer.step()
        self.policy_scheduler.step()
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

    def simulate_train(self, episodes=1, max_steps=50):
        """
        Run training simulation for specified number of episodes.
        This method is called by MultiAgentTrainer to train the agent.
        
        Args:
            episodes: Number of episodes to run
            max_steps: Maximum steps per episode
            
        Returns:
            dict: Training results including rewards, steps, etc.
        """
        total_reward = 0.0
        total_steps = 0
        episode_rewards = []
        
        for episode in range(episodes):
            episode_reward = 0.0
            episode_steps = 0
            
            # Reset environment for new episode
            if hasattr(self.env, "reset"):
                self.env.reset()
            
            # Run episode steps
            for step in range(1, max_steps + 1):
                try:
                    # Get action from simulate_step
                    step_result = self.simulate_step(episode + 1, step)
                    
                    if step_result and isinstance(step_result, dict):
                        step_reward = step_result.get("reward", 0.0)
                        episode_reward += step_reward
                        total_reward += step_reward
                    
                    episode_steps += 1
                    total_steps += 1
                    
                    # Check if episode should end early
                    if step_result and step_result.get("done", False):
                        break
                        
                except Exception as e:
                    console.print(f"[yellow]⚠ BlueAgent training step error: {e}[/yellow]")
                    break
            
            episode_rewards.append(episode_reward)
            
            # Perform batch training if we have enough experiences
            if len(self.replay_buffer.buffer) >= self.batch_size:
                try:
                    self.train_on_batch()
                except Exception as e:
                    console.print(f"[yellow]⚠ BlueAgent batch training error: {e}[/yellow]")
        
        # Update episode counter
        self.total_episodes += episodes
        
        # Return training results in expected format
        return {
            "agent_id": self.agent_id,
            "episodes": episodes,
            "total_reward": total_reward,
            "average_reward": total_reward / episodes if episodes > 0 else 0.0,
            "total_steps": total_steps,
            "episode_rewards": episode_rewards,
            "final_epsilon": float(self.epsilon),
            "replay_buffer_size": len(self.replay_buffer.buffer) if hasattr(self.replay_buffer, 'buffer') else 0,
            "blue_team_alert": getattr(self, '_last_state', {}).get("blue_team_alert", 0.0),
            "gpt_calls": self.stats_monitor._get_agent_stats(self.agent_id).gpt_calls if self.agent_id in self.stats_monitor.agent_stats else 0,
            "current_mode": getattr(self, 'current_mode', 'Standard')
        }

    def _log_training_event(self, msg):
        with open(self.training_log_path, "a") as f:
            f.write(f"{msg}\n")

    @staticmethod
    def encode_env_state_static(state, device, **kwargs):
        """Encode environment state into a rich 512-dim feature vector.
        
        Phase 3: Uses the shared state_encoder module with 90+ meaningful
        dimensions instead of the legacy 19-dim encoding.
        """
        from core.models.state_encoder import encode_state
        return encode_state(state, device, **kwargs)

    def encode_env_state(self, state, **kwargs):
        """Instance method for state encoding with agent context."""
        state = dict(state)
        if hasattr(self, "last_action") and self.last_action is not None:
            state["last_action"] = self.last_action
        encoding_kwargs = {
            "current_step": getattr(self, 'total_steps', 0) % 100,
            "max_steps": 100,
        }
        encoding_kwargs.update(kwargs)
        return self.encode_env_state_static(state, self.device, **encoding_kwargs)

    def sync_memory(self):
        try:
            if self.memory_router:
                self.memory_router.sync_global_insights()
            from core.logic.redundancy_detector import detect_redundancy_batch
            self.replay_buffer.prune_redundancy(lambda cmds: detect_redundancy_batch(cmds))
            return True
        except Exception as e:
            console.print(f"[yellow]Warning: BlueAgent memory sync failed: {e}[/yellow]")
            return False

    def get_smart_command(self, state, phase, cache_key=None):
        """
        Use GPTManager for command generation.
        Integrate enhanced defensive strategy analysis.
        """
        task_desc = (
            f"Phase: {phase}, Privilege: {state.get('privilege_level')}, "
            f"Alert: {state.get('blue_team_alert')}, Risk: {state.get('detection_risk')}."
            " Suggest the most effective, non-redundant, phase-appropriate defensive command."
        )
        # Use smart_decision for complex tactical decisions
        dual_feedback = self.gpt_manager.smart_decision(
            task_type="complex_defense", 
            task_description=task_desc
        )
        command = self.gpt_manager._sanitize_output(dual_feedback)
        # Store feedback for learning and adaptation
        if hasattr(self, "gpt_reasoning_cache"):
            self.gpt_reasoning_cache[f"tactical_feedback_{phase}"] = dual_feedback
        return command

    def reset(self):
        self.total_steps = 0
        self.command_history.clear()
        self.prioritized_experiences.clear()
        self.prioritized_priorities.clear()
        if hasattr(self, "stats_monitor"):
            self.stats_monitor.reset()
        if hasattr(self, "gpt_reasoning_cache") and "tactical_feedback_reset" in self.gpt_reasoning_cache:
            feedback = self.gpt_reasoning_cache["tactical_feedback_reset"]
            # Log or use as needed for strategy adaptation

    def display_advanced_status(self):
        """Display advanced status information for the BlueAgent."""
        console.print(Panel.fit(
            f"[bold cyan]BlueAgent Advanced Status[/bold cyan]\n"
            f"Mode: {self.current_mode}\n"
            f"Epsilon: {self.epsilon:.3f}\n"
            f"Total Steps: {self.total_steps}\n"
            f"Total Episodes: {self.total_episodes}\n"
            f"Command History: {len(self.command_history)} commands\n"
            f"Memory Router: {'Connected' if self.memory_router else 'Not connected'}\n"
            f"Last Action: {self.last_action}\n"
            f"Repeat Steps: {self.repeat_steps}",
            title="[bold blue]BlueAgent Status[/bold blue]"
        ))
        
    def generate_chain_snapshot(self):
        """Generate a snapshot of the current action chain."""
        console.print(f"[blue]BlueAgent: Generating chain snapshot...[/blue]")
        snapshot = {
            "agent_id": self.agent_id,
            "command_history": self.command_history[-10:],  # Last 10 commands
            "current_mode": self.current_mode,
            "total_steps": self.total_steps,
            "timestamp": time.time()
        }
        return snapshot
        
    def safe_shutdown(self):
        """Safely shutdown the BlueAgent."""
        console.print(f"[blue]BlueAgent: Shutting down safely...[/blue]")
        try:
            # Save any important state and display final stats
            console.print(f"[cyan]BlueAgent final stats: {self.total_steps} steps, {self.total_episodes} episodes[/cyan]")
            
            # Sync memory one last time
            if self.memory_router:
                self.sync_memory()
                
            console.print(f"[green]BlueAgent shutdown complete.[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning during BlueAgent shutdown: {e}[/yellow]")
    
    def provide_reasoning(self, context_type: str, context_data: dict) -> str:
        """Provide reasoning for a given context - only available in OrionAgent."""
        return f"BlueAgent does not provide strategic reasoning. Use OrionAgent for strategic insights."

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
    agent.display_advanced_status()
    agent.generate_chain_snapshot()

    agent.safe_shutdown()
