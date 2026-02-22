"""core/algorithms/options_hrl.py — Phase 49: Options Framework / HRL.

Hierarchical Reinforcement Learning via the Options framework.
Defines reusable multi-step "skills" (options) as temporal abstractions
that compose sequences of primitive actions into macro-behaviours.

Key concepts:
  - **Option**: (initiation_set, policy, termination_condition)
  - **OptionPolicy**: selects among available options at the high level
  - **Intra-option policy**: selects primitive actions within an option
  - **Termination**: learned or rule-based condition to end an option

Pre-defined pentesting skills:
  - full_port_scan: nmap -sC -sV + targeted scans → 3-5 steps
  - web_enum: gobuster + nikto + curl → 3-4 steps
  - exploit_chain: searchsploit + msfconsole + verify → 2-4 steps
  - privesc_routine: linpeas + sudo -l + SUID search → 3 steps
  - exfil_chain: find flag + encode + transfer → 3 steps

Architecture:
    HighLevelPolicy → select_option()
    ↓
    Option(policy_π, β_termination)
    ↓
    Primitive actions via PPO / registry
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("ariaska.algorithms.options_hrl")


@dataclass
class OptionConfig:
    """Configuration for a single option (skill)."""
    name: str
    description: str = ""
    max_steps: int = 10
    # Which attack phases this option is valid in
    valid_phases: List[str] = field(default_factory=list)
    # Command templates that comprise this skill
    command_templates: List[str] = field(default_factory=list)
    # Preconditions (state keys that must be truthy)
    preconditions: List[str] = field(default_factory=list)
    # Reward bonus for completing the full option
    completion_bonus: float = 5.0


@dataclass
class HRLConfig:
    """Configuration for the Options/HRL framework.

    Args:
        enabled: Master switch.
        state_dim: Input state dimension.
        hidden_dim: Hidden layer dimension for option policies.
        n_primitive_actions: Number of primitive actions (PPO action_dim).
        termination_reg: Regularization to discourage premature termination.
        interest_threshold: Minimum interest probability to consider an option.
        deliberation_cost: Small penalty per option switch to encourage commitment.
    """
    enabled: bool = True
    state_dim: int = 512
    hidden_dim: int = 128
    n_primitive_actions: int = 5
    termination_reg: float = 0.01
    interest_threshold: float = 0.3
    deliberation_cost: float = 0.1


class Option:
    """A single temporal abstraction (skill) in the Options framework.

    Each option has:
      - An initiation check (can this option start from current state?)
      - An intra-option policy (which primitives to execute)
      - A termination condition (when to stop and return to high-level)
    """

    def __init__(self, config: OptionConfig) -> None:
        self.config = config
        self.name = config.name
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._active: bool = False
        self._command_idx: int = 0

    def can_initiate(self, state: Dict[str, Any]) -> bool:
        """Check if this option can start from the current state."""
        phase = state.get("phase", "RECON").upper()
        if self.config.valid_phases and phase not in [
            p.upper() for p in self.config.valid_phases
        ]:
            return False
        for precond in self.config.preconditions:
            if not state.get(precond, False):
                return False
        return True

    def start(self) -> None:
        """Begin executing this option."""
        self._step_count = 0
        self._total_reward = 0.0
        self._active = True
        self._command_idx = 0
        logger.debug(f"Option '{self.name}' started")

    def next_command(self) -> Optional[str]:
        """Get next command template in the skill sequence.

        Returns:
            Command template string, or None if exhausted.
        """
        if self._command_idx >= len(self.config.command_templates):
            return None
        cmd = self.config.command_templates[self._command_idx]
        self._command_idx += 1
        self._step_count += 1
        return cmd

    def should_terminate(self, state: Dict[str, Any], reward: float) -> bool:
        """Check if this option should terminate.

        Terminates when:
          - All commands in the sequence are exhausted
          - Max steps exceeded
          - A discovery triggers completion
          - Negative reward streak suggests failure
        """
        self._total_reward += reward

        # Exhausted command sequence
        if self._command_idx >= len(self.config.command_templates):
            return True

        # Max steps
        if self._step_count >= self.config.max_steps:
            return True

        # Early termination on repeated failure
        if self._step_count >= 3 and self._total_reward < -5.0:
            return True

        return False

    def terminate(self) -> Dict[str, Any]:
        """End this option and return summary."""
        self._active = False
        summary = {
            "option": self.name,
            "steps_taken": self._step_count,
            "total_reward": self._total_reward,
            "completed_sequence": self._command_idx >= len(self.config.command_templates),
        }
        logger.debug(
            f"Option '{self.name}' terminated: "
            f"steps={self._step_count}, reward={self._total_reward:.2f}"
        )
        return summary

    @property
    def is_active(self) -> bool:
        """Whether this option is currently executing."""
        return self._active


class OptionTermination(nn.Module):
    """Learned termination function β(s) for an option.

    Outputs probability of terminating given current state.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns termination probability β(s) in [0, 1]."""
        return self.net(state).squeeze(-1)


class HighLevelPolicy(nn.Module):
    """Policy over options — selects which skill to execute.

    Architecture:
        state (512) → Linear → ReLU → Linear → softmax → option probabilities
    """

    def __init__(
        self,
        state_dim: int,
        n_options: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.n_options = n_options
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_options),
        )
        # Small init
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns logits over options (B, n_options)."""
        return self.net(state)

    def select_option(
        self,
        state: torch.Tensor,
        available_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor, float]:
        """Select an option given state and availability mask.

        Args:
            state: (state_dim,) or (1, state_dim) state tensor.
            available_mask: (n_options,) bool mask of valid options.
            deterministic: If True, pick argmax.

        Returns:
            (option_idx, log_prob, entropy)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        logits = self.forward(state).squeeze(0)  # (n_options,)

        # Mask unavailable options
        if available_mask is not None:
            logits = logits.masked_fill(~available_mask, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)

        if deterministic:
            option_idx = probs.argmax().item()
        else:
            option_idx = dist.sample().item()

        log_prob = dist.log_prob(torch.tensor(option_idx))
        entropy = dist.entropy().item()

        return option_idx, log_prob, entropy


class OptionsManager:
    """Manages a collection of options and the high-level policy.

    Usage::

        manager = OptionsManager(config)
        manager.register_default_pentesting_options()

        # At decision time:
        option = manager.select_option(state_dict, state_tensor)
        while option.is_active:
            cmd = option.next_command()
            # execute cmd, get reward
            if option.should_terminate(state_dict, reward):
                summary = option.terminate()
                break

        stats = manager.get_stats()
    """

    def __init__(self, config: Optional[HRLConfig] = None) -> None:
        self.config = config or HRLConfig()
        self._options: List[Option] = []
        self._option_map: Dict[str, int] = {}
        self._high_level_policy: Optional[HighLevelPolicy] = None
        self._terminations: List[Optional[OptionTermination]] = []
        # Tracking
        self._option_usage: Dict[str, int] = {}
        self._option_success: Dict[str, int] = {}
        self._active_option: Optional[Option] = None

    def register_option(self, config: OptionConfig) -> None:
        """Register a new option (skill)."""
        idx = len(self._options)
        option = Option(config)
        self._options.append(option)
        self._option_map[config.name] = idx
        self._option_usage[config.name] = 0
        self._option_success[config.name] = 0
        # Add learned termination
        self._terminations.append(
            OptionTermination(self.config.state_dim)
        )
        logger.debug(f"Registered option '{config.name}' at index {idx}")

    def register_default_pentesting_options(self) -> None:
        """Register pre-built pentesting skill options."""
        defaults = [
            OptionConfig(
                name="full_port_scan",
                description="Comprehensive port scanning and service detection",
                max_steps=5,
                valid_phases=["RECON", "ENUMERATION"],
                command_templates=[
                    "nmap -sC -sV -p- {target}",
                    "nmap -sU --top-ports 50 {target}",
                    "nmap -sV --version-intensity 5 -p {open_ports} {target}",
                ],
                preconditions=[],
                completion_bonus=8.0,
            ),
            OptionConfig(
                name="web_enum",
                description="Web application enumeration",
                max_steps=5,
                valid_phases=["ENUMERATION", "EXPLOITATION"],
                command_templates=[
                    "gobuster dir -u http://{target} -w /usr/share/wordlists/dirb/common.txt",
                    "nikto -h http://{target}",
                    "curl -s http://{target}/robots.txt",
                ],
                preconditions=["ports_discovered"],
                completion_bonus=6.0,
            ),
            OptionConfig(
                name="exploit_chain",
                description="Exploit search and execution chain",
                max_steps=6,
                valid_phases=["EXPLOITATION"],
                command_templates=[
                    "searchsploit {service} {version}",
                    "msfconsole -q -x 'search type:exploit {service}; exit'",
                ],
                preconditions=["ports_discovered"],
                completion_bonus=15.0,
            ),
            OptionConfig(
                name="privesc_routine",
                description="Privilege escalation enumeration",
                max_steps=5,
                valid_phases=["PRIVILEGE_ESCALATION"],
                command_templates=[
                    "sudo -l",
                    "find / -perm -4000 -type f 2>/dev/null",
                    "cat /etc/crontab",
                ],
                preconditions=["shell_obtained"],
                completion_bonus=12.0,
            ),
            OptionConfig(
                name="credential_harvest",
                description="Credential hunting and extraction",
                max_steps=5,
                valid_phases=["ENUMERATION", "EXPLOITATION", "POST_EXPLOITATION"],
                command_templates=[
                    "find / -name '*.conf' -o -name '*.cfg' -o -name '*.ini' 2>/dev/null | head -20",
                    "cat /etc/shadow 2>/dev/null || cat /etc/passwd",
                    "grep -r 'password\\|passwd\\|pwd' /var/www/ /opt/ /home/ 2>/dev/null | head -20",
                ],
                preconditions=["shell_obtained"],
                completion_bonus=10.0,
            ),
            OptionConfig(
                name="exfil_chain",
                description="Data exfiltration sequence",
                max_steps=5,
                valid_phases=["EXFILTRATION", "POST_EXPLOITATION"],
                command_templates=[
                    "find / -name 'flag*' -o -name 'root.txt' -o -name 'user.txt' 2>/dev/null",
                    "cat {flag_path}",
                    "base64 {flag_path}",
                ],
                preconditions=["shell_obtained"],
                completion_bonus=20.0,
            ),
        ]
        for opt_cfg in defaults:
            self.register_option(opt_cfg)

        # Build high-level policy now that we know n_options
        self._build_high_level_policy()

    def _build_high_level_policy(self) -> None:
        """Build the high-level policy network."""
        if self._options:
            self._high_level_policy = HighLevelPolicy(
                state_dim=self.config.state_dim,
                n_options=len(self._options),
                hidden_dim=self.config.hidden_dim,
            )

    def select_option(
        self,
        state_dict: Dict[str, Any],
        state_tensor: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Optional[Option]:
        """Select the best option to execute given current state.

        Args:
            state_dict: Environment state dictionary.
            state_tensor: Optional state tensor for neural selection.
            deterministic: If True, pick highest-probability option.

        Returns:
            Selected Option, or None if no valid options.
        """
        # Find available options
        available = []
        for i, opt in enumerate(self._options):
            if opt.can_initiate(state_dict) and not opt.is_active:
                available.append(i)

        if not available:
            return None

        # Neural selection if we have a state tensor and policy
        if state_tensor is not None and self._high_level_policy is not None:
            mask = torch.zeros(len(self._options), dtype=torch.bool)
            for i in available:
                mask[i] = True

            option_idx, log_prob, entropy = self._high_level_policy.select_option(
                state_tensor, available_mask=mask, deterministic=deterministic,
            )
        else:
            # Fallback: pick based on usage frequency (prefer under-used)
            import random
            usage_counts = [self._option_usage.get(self._options[i].name, 0) for i in available]
            min_usage = min(usage_counts) if usage_counts else 0
            least_used = [
                available[j] for j, c in enumerate(usage_counts) if c == min_usage
            ]
            option_idx = random.choice(least_used)

        option = self._options[option_idx]
        option.start()
        self._active_option = option
        self._option_usage[option.name] = self._option_usage.get(option.name, 0) + 1

        logger.info(f"[HRL] Selected option '{option.name}' (idx={option_idx})")
        return option

    @property
    def active_option(self) -> Optional[Option]:
        """Currently executing option, if any."""
        return self._active_option

    def step(
        self,
        state_dict: Dict[str, Any],
        reward: float,
    ) -> Tuple[Optional[str], bool]:
        """Execute one step of the active option.

        Args:
            state_dict: Current environment state.
            reward: Reward from last primitive action.

        Returns:
            (next_command, option_terminated)
        """
        if self._active_option is None:
            return None, True

        opt = self._active_option

        # Check termination
        if opt.should_terminate(state_dict, reward):
            summary = opt.terminate()
            if summary.get("completed_sequence", False):
                self._option_success[opt.name] = (
                    self._option_success.get(opt.name, 0) + 1
                )
            self._active_option = None
            return None, True

        # Get next command
        cmd = opt.next_command()
        if cmd is None:
            summary = opt.terminate()
            self._active_option = None
            return None, True

        return cmd, False

    def get_stats(self) -> Dict[str, Any]:
        """Return HRL framework statistics."""
        return {
            "n_options": len(self._options),
            "option_names": [o.name for o in self._options],
            "usage": dict(self._option_usage),
            "success": dict(self._option_success),
            "active_option": self._active_option.name if self._active_option else None,
            "has_high_level_policy": self._high_level_policy is not None,
        }

    def reset(self) -> None:
        """Reset active option state (keep registrations)."""
        if self._active_option and self._active_option.is_active:
            self._active_option.terminate()
        self._active_option = None
        for opt in self._options:
            if opt.is_active:
                opt.terminate()
