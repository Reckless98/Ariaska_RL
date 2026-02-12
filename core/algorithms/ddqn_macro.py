#!/usr/bin/env python3
"""
core/algorithms/ddqn_macro.py — ARIASKA Hierarchical DDQN Macro-Intent Selector v1.0
🧠 Double DQN | 🎯 Macro-Intent Options | 📊 Target Network | 🔬 Experience Replay

High-level option selector for the hierarchical RL system:
  - DDQN selects macro-intents (RECON_FOCUS, CREDENTIAL_CHAIN, etc.)
  - PPO selects concrete commands consistent with chosen macro-intent
  - Off-policy learning from PPO rollouts
  - Option-level reward shaping for clean phase transitions

Macro-intents are abstract strategic goals that constrain PPO's action space:
  - Each macro-intent maps to a subset of command templates
  - DDQN learns which macro-intent to pursue based on current state
  - Supports forward phase progression bias

Usage in SmartCoach:
    ddqn = DDQNMacro(state_dim=512, device="cpu")
    macro = ddqn.select_macro(state_tensor, phase)
    # PPO selects command from macro.allowed_commands
    ddqn.store(state, macro_idx, reward, next_state, done)
    ddqn.update()  # Off-policy Double DQN update
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import logging
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger("ariaska.ddqn_macro")


# ═══════════════════════════════════════════════════════════════════════════════
# MACRO-INTENT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class MacroIntent(IntEnum):
    """High-level strategic options for the hierarchical RL system.
    
    Each macro-intent represents a strategic goal that constrains which
    commands PPO can select. This creates a natural hierarchy:
    DDQN picks strategy → PPO picks tactics.
    """
    RECON_FOCUS = 0        # Network scanning, port discovery, service fingerprinting
    SERVICE_ENUM = 1       # Deep service enumeration (SMB, FTP, SNMP, etc.)
    CREDENTIAL_CHAIN = 2   # Brute force, default creds, credential reuse
    SERVICE_EXPLOIT = 3    # Known vulnerability exploitation (CVEs, backdoors)
    POST_SHELL_ENUM = 4    # Post-exploitation enumeration (privesc vectors)
    PRIV_ESC = 5           # Privilege escalation attempts
    LATERAL = 6            # Lateral movement to other services/hosts
    EXFIL = 7              # Data exfiltration (shadow, passwd, keys)
    CLOSEOUT = 8           # Anti-forensics, cleanup, session closure


NUM_MACROS = len(MacroIntent)


# ═══════════════════════════════════════════════════════════════════════════════
# MACRO-INTENT → COMMAND TEMPLATE MAPPING
# Each macro-intent allows a subset of command templates from the registry.
# PPO's action mask is further constrained to only these templates.
# ═══════════════════════════════════════════════════════════════════════════════

MACRO_COMMAND_MAP: Dict[MacroIntent, Set[str]] = {
    MacroIntent.RECON_FOCUS: {
        "nmap_quick_scan", "nmap_top_ports", "nmap_full_tcp", "nmap_service_version",
        "nmap_vuln_scan", "nmap_udp_scan", "nmap_os_detection", "masscan_fast",
        "nmap_stealth_scan", "nmap_comprehensive", "nmap_port_21", "nmap_port_1524",
        "nmap_port_8180", "nmap_port_5432", "nmap_port_6667", "nmap_rservices",
        "dig_any", "dig_axfr", "dnsrecon", "dns_enum", "whois_lookup",
        "whatweb", "curl_headers", "rpcinfo_check", "showmount_enum",
    },
    MacroIntent.SERVICE_ENUM: {
        "snmpwalk", "onesixtyone", "showmount", "nfs_mount",
        "smbclient_list", "smbclient_connect", "smbclient_auth",
        "smbmap_shares", "enum4linux_full", "enum4linux_ng", "enum4linux_scan",
        "rpcclient_null", "rpcclient_enumdomusers",
        "ssh_audit", "nikto_scan", "nuclei_scan",
        "ldapsearch_base", "ldapsearch_users",
        "ftp_anonymous", "redis_cli", "wpscan",
    },
    MacroIntent.CREDENTIAL_CHAIN: {
        "hydra_ssh", "hydra_ftp", "hydra_smb", "hydra_http_form",
        "crackmapexec_smb_bruteforce", "cme_smb_bruteforce",
        "ssh_login", "mysql_login", "mssql_login",
        "ftp_anonymous", "psql_default_creds", "tomcat_cred_test",
        "mysql_root_login",
    },
    MacroIntent.SERVICE_EXPLOIT: {
        "msfconsole_exploit", "msfconsole_auto", "searchsploit",
        "vsftpd_exploit", "unrealircd_exploit", "samba_exploit",
        "java_rmi_exploit", "distcc_exploit", "psql_rce",
        "telnet_1524", "rsh_root", "rlogin_root",
        "sqlmap_get", "sqlmap_post", "sqlmap_test", "sqlmap_shell",
        "war_deploy",
        "impacket_psexec", "impacket_smbexec", "impacket_wmiexec",
    },
    MacroIntent.POST_SHELL_ENUM: {
        "sudo_list", "find_suid", "find_capabilities", "find_writable_etc",
        "kernel_exploit_check", "sudo_check", "find_sgid",
        "linpeas", "linpeas_local", "pspy", "pspy_monitor",
        "cron_check", "writable_etc_passwd", "capability_check",
        "whoami_all", "systeminfo",
    },
    MacroIntent.PRIV_ESC: {
        "sudo_list", "find_suid", "find_capabilities",
        "kernel_exploit_check", "docker_privesc", "lxd_privesc",
        "winpeas", "powerup", "windows_exploit_suggester",
        "accesschk_services",
        "root_shell_confirm",
    },
    MacroIntent.LATERAL: {
        "ssh_lateral", "ssh_tunnel_local", "ssh_tunnel_dynamic",
        "chisel_server", "chisel_client", "ligolo_agent",
        "impacket_pth_psexec", "crackmapexec_pth",
        "evil_winrm", "evil_winrm_hash", "winrm_exec",
        "responder", "ntlmrelayx",
    },
    MacroIntent.EXFIL: {
        "dump_shadow", "dump_passwd", "credential_dump", "hashdump",
        "impacket_secretsdump", "secretsdump_dc",
        "mimikatz_logonpasswords", "mimikatz_sam", "mimikatz_dcsync",
    },
    MacroIntent.CLOSEOUT: {
        # Anti-forensics and cleanup commands
        # These are typically generated by the closeout gate in SmartCoach
        "shred_logs", "timestomp", "clear_history", "remove_artifacts",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE → ALLOWED MACROS
# Restricts which macro-intents are valid for each attack phase.
# Enforces forward progression with limited one-phase-behind allowance.
# ═══════════════════════════════════════════════════════════════════════════════

def _get_phase_ordinal(phase_name: str) -> int:
    """Map phase name to ordinal for comparison."""
    _PHASE_ORDER = {
        "RECON": 0, "ENUMERATION": 1, "EXPLOITATION": 2,
        "PRIVILEGE_ESCALATION": 3, "LATERAL_MOVEMENT": 4,
        "POST_EXPLOITATION": 5, "EXFILTRATION": 6, "CLOSEOUT": 7,
    }
    return _PHASE_ORDER.get(phase_name, 0)


# Which macros are appropriate per phase (forward-biased)
PHASE_MACRO_ALLOWLIST: Dict[str, List[MacroIntent]] = {
    "RECON": [
        MacroIntent.RECON_FOCUS, MacroIntent.SERVICE_ENUM,
    ],
    "ENUMERATION": [
        MacroIntent.RECON_FOCUS, MacroIntent.SERVICE_ENUM,
        MacroIntent.CREDENTIAL_CHAIN,
    ],
    "EXPLOITATION": [
        MacroIntent.SERVICE_ENUM, MacroIntent.CREDENTIAL_CHAIN,
        MacroIntent.SERVICE_EXPLOIT,
    ],
    "PRIVILEGE_ESCALATION": [
        MacroIntent.SERVICE_EXPLOIT, MacroIntent.POST_SHELL_ENUM,
        MacroIntent.PRIV_ESC,
    ],
    "LATERAL_MOVEMENT": [
        MacroIntent.POST_SHELL_ENUM, MacroIntent.PRIV_ESC,
        MacroIntent.LATERAL,
    ],
    "POST_EXPLOITATION": [
        MacroIntent.PRIV_ESC, MacroIntent.LATERAL,
        MacroIntent.EXFIL,
    ],
    "EXFILTRATION": [
        MacroIntent.EXFIL, MacroIntent.LATERAL,
        MacroIntent.CLOSEOUT,
    ],
    "CLOSEOUT": [
        MacroIntent.CLOSEOUT,
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# DDQN CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DDQNConfig:
    """Configuration for the DDQN macro-intent selector."""
    state_dim: int = 512
    num_macros: int = NUM_MACROS
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    # Double DQN
    learning_rate: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005               # Soft target update rate
    target_update_freq: int = 50     # Hard target update every N steps (if not using soft)
    # Exploration
    # R42: Reduced epsilon_decay from 2000→300 for faster convergence in 10-episode runs.
    # At 160 steps/run, old decay only reached ε≈0.48. New decay reaches ε≈0.26,
    # enabling DDQN_UNCERTAINTY mentor trigger at [0.15, 0.45] to fire by mid-run.
    epsilon_start: float = 0.5       # Initial exploration
    epsilon_end: float = 0.10        # R43: 0.05→0.10 — prevents over-exploitation in longer runs
    epsilon_decay: int = 300         # Steps to decay epsilon (was 2000)
    # Replay buffer
    buffer_size: int = 10000
    batch_size: int = 64
    min_buffer_size: int = 128       # Don't train until this many samples
    # Training
    update_freq: int = 4             # Update every N environment steps
    grad_clip: float = 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# REPLAY BUFFER
# ═══════════════════════════════════════════════════════════════════════════════

class MacroReplayBuffer:
    """Experience replay buffer for DDQN macro-intent transitions.
    
    Stores (state, macro_intent, reward, next_state, done) tuples.
    Supports uniform random sampling for off-policy learning.
    """

    def __init__(self, capacity: int = 10000):
        self.buffer: deque = deque(maxlen=capacity)

    def add(
        self,
        state: torch.Tensor,
        macro: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ):
        """Store a macro-level transition."""
        self.buffer.append((
            state.detach().cpu(),
            macro,
            reward,
            next_state.detach().cpu(),
            done,
        ))

    def sample(self, batch_size: int) -> Tuple:
        """Sample a random batch of transitions."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, macros, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(macros, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════════════════════════
# DDQN NETWORK
# ═══════════════════════════════════════════════════════════════════════════════

class MacroQNetwork(nn.Module):
    """Q-Network for macro-intent selection.
    
    Dueling architecture with separate value and advantage streams
    for more stable Q-value estimation.
    
    Architecture:
        Input (512) → Shared Backbone → ├─ Value Stream → V(s)
                                        └─ Advantage Stream → A(s, a)
        Q(s, a) = V(s) + A(s, a) - mean(A(s, :))
    """

    def __init__(self, config: DDQNConfig):
        super().__init__()
        dims = config.hidden_dims

        # Shared feature extraction
        layers = []
        in_dim = config.state_dim
        for d in dims:
            layers.extend([
                nn.Linear(in_dim, d),
                nn.LayerNorm(d),
                nn.GELU(),
                nn.Dropout(0.05),
            ])
            in_dim = d
        self.backbone = nn.Sequential(*layers)

        final_dim = dims[-1]

        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.GELU(),
            nn.Linear(final_dim // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.GELU(),
            nn.Linear(final_dim // 2, config.num_macros),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for stable Q-learning."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q-values for all macro-intents.
        
        Args:
            state: (B, state_dim) tensor.
            
        Returns:
            Q-values: (B, num_macros) tensor.
        """
        features = self.backbone(state)
        value = self.value_stream(features)            # (B, 1)
        advantage = self.advantage_stream(features)    # (B, num_macros)
        # Dueling: Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values


# ═══════════════════════════════════════════════════════════════════════════════
# DDQN MACRO-INTENT SELECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class DDQNMacro:
    """Double DQN macro-intent selector for hierarchical RL.
    
    Selects high-level strategic options (macro-intents) that constrain
    PPO's low-level action selection. Uses:
      - Dueling Double DQN with target network
      - Experience replay buffer
      - Phase-aware macro masking
      - Epsilon-greedy exploration with decay
      - Option-level reward shaping
    
    Usage:
        ddqn = DDQNMacro(config, device="cpu")
        macro, q_values, confidence = ddqn.select_macro(state, phase_name)
        # ... PPO selects command from macro's allowed set ...
        ddqn.store_transition(state, macro, reward, next_state, done)
        loss = ddqn.update()
    """

    def __init__(
        self,
        config: Optional[DDQNConfig] = None,
        device: str = "cpu",
    ):
        self.config = config or DDQNConfig()
        self.device = torch.device(device)

        # Online and target networks
        self.online_net = MacroQNetwork(self.config).to(self.device)
        self.target_net = MacroQNetwork(self.config).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(),
            lr=self.config.learning_rate,
        )

        # Replay buffer
        self.replay_buffer = MacroReplayBuffer(self.config.buffer_size)

        # Exploration
        self.epsilon = self.config.epsilon_start
        self.total_steps = 0
        self.update_count = 0

        # Metrics
        self.training_metrics: Dict[str, List[float]] = {
            "loss": [],
            "q_values": [],
            "epsilon": [],
            "td_errors": [],
            "macro_distribution": [],
        }

        # Track current episode macro history for reward shaping
        self._episode_macros: List[int] = []
        self._last_macro: Optional[int] = None

        # ── R57 Layer 1: Macro persistence & stagnation detection ──
        self._macro_hold_counter: int = 0            # Steps current macro has been held
        self._macro_min_hold: int = 3                # Minimum steps before macro switch allowed
        self._stagnation_counter: int = 0            # Steps in same phase without discoveries
        self._stagnation_threshold: int = 10         # Trigger stagnation override after N steps
        self._last_phase_name: Optional[str] = None  # Track phase for stagnation
        self._per_macro_success: Dict[int, Dict[str, float]] = {}  # macro_idx → {hits, total, reward_sum}

        logger.info(
            f"DDQNMacro initialized: state_dim={self.config.state_dim}, "
            f"num_macros={self.config.num_macros}, "
            f"params={sum(p.numel() for p in self.online_net.parameters()):,}"
        )

    def select_macro(
        self,
        state: torch.Tensor,
        phase_name: str = "RECON",
        force_macro: Optional[MacroIntent] = None,
        had_discovery: bool = False,
    ) -> Tuple[MacroIntent, torch.Tensor, float]:
        """Select a macro-intent using epsilon-greedy Double DQN.
        
        R57 Layer 1 enhancements:
          - Macro persistence: hold current macro for min 3 steps before switching
          - Stagnation detection: force forward macro after 10 steps without discoveries
          - Per-macro success tracking for future biasing
        
        Args:
            state: (state_dim,) state tensor.
            phase_name: Current attack phase name for masking.
            force_macro: If set, force this macro-intent (for testing/mentor override).
            had_discovery: Whether the previous step produced any new discoveries.
            
        Returns:
            (macro_intent, q_values, confidence) where confidence is
            the normalized Q-value separation between best and second-best.
        """
        # ── R57: Update stagnation counter ──
        if phase_name == self._last_phase_name:
            if had_discovery:
                self._stagnation_counter = 0  # Reset on discovery
            else:
                self._stagnation_counter += 1
        else:
            # Phase changed — reset stagnation
            self._stagnation_counter = 0
        self._last_phase_name = phase_name

        if force_macro is not None:
            # Return forced macro with dummy Q-values
            q_dummy = torch.zeros(self.config.num_macros)
            q_dummy[force_macro.value] = 1.0
            self._macro_hold_counter = 0
            self._last_macro = force_macro.value
            self._episode_macros.append(force_macro.value)
            self.total_steps += 1
            self._decay_epsilon()
            return force_macro, q_dummy, 1.0

        # Get phase-valid macro mask
        allowed = PHASE_MACRO_ALLOWLIST.get(phase_name, list(MacroIntent))
        macro_mask = torch.zeros(self.config.num_macros, dtype=torch.bool)
        for m in allowed:
            macro_mask[m.value] = True

        # Ensure at least one macro is allowed
        if not macro_mask.any():
            macro_mask[:] = True

        with torch.no_grad():
            state_t = state.unsqueeze(0).to(self.device) if state.dim() == 1 else state.to(self.device)
            q_values = self.online_net(state_t).squeeze(0)  # (num_macros,)

        # Apply mask: invalid macros get Q = -inf
        masked_q = q_values.clone()
        masked_q[~macro_mask] = float("-inf")

        # ── R57: Stagnation override — force most forward macro ──
        stagnation_override = False
        if self._stagnation_counter >= self._stagnation_threshold:
            forward_macro = _get_stagnation_override(phase_name, allowed)
            if forward_macro is not None and macro_mask[forward_macro.value]:
                macro_idx = forward_macro.value
                stagnation_override = True
                logger.info(
                    f"[DDQN-STAGNATION] Forcing {forward_macro.name} after "
                    f"{self._stagnation_counter} stagnant steps in {phase_name}"
                )
                self._stagnation_counter = 0  # Reset after override

        if not stagnation_override:
            # ── R57: Macro persistence — hold for min steps ──
            if (
                self._last_macro is not None
                and self._macro_hold_counter < self._macro_min_hold
                and macro_mask[self._last_macro]  # Still valid in current phase
            ):
                macro_idx = self._last_macro
                # Don't increment here — tracking block below handles it
            else:
                # Epsilon-greedy selection
                if random.random() < self.epsilon:
                    # Random from allowed macros
                    valid_indices = macro_mask.nonzero(as_tuple=True)[0].tolist()
                    macro_idx = random.choice(valid_indices)
                else:
                    # Greedy
                    macro_idx = masked_q.argmax().item()

        # Compute confidence: Q-separation between best and second-best
        sorted_q = masked_q[macro_mask].sort(descending=True).values
        if len(sorted_q) >= 2:
            q_spread = (sorted_q[0] - sorted_q[1]).item()
            confidence = min(1.0, max(0.0, q_spread / (abs(sorted_q[0].item()) + 1e-6)))
        else:
            confidence = 1.0

        macro = MacroIntent(macro_idx)

        # ── R57: Track macro hold / switch ──
        if self._last_macro is not None and macro_idx != self._last_macro:
            self._macro_hold_counter = 1  # New macro — start hold
        elif self._last_macro is None:
            self._macro_hold_counter = 1  # First macro of episode
        else:
            self._macro_hold_counter += 1  # Same macro — increment hold

        # Track
        self._episode_macros.append(macro_idx)
        self._last_macro = macro_idx
        self.total_steps += 1

        # Decay epsilon
        self._decay_epsilon()

        return macro, q_values.cpu(), confidence

    def _decay_epsilon(self):
        """Decay epsilon toward minimum."""
        self.epsilon = max(
            self.config.epsilon_end,
            self.config.epsilon_start - (
                self.total_steps / self.config.epsilon_decay
            ) * (self.config.epsilon_start - self.config.epsilon_end),
        )

    def record_macro_outcome(self, macro_idx: int, reward: float, had_discovery: bool):
        """R57 Layer 1: Track per-macro success for future biasing.
        
        Args:
            macro_idx: MacroIntent index.
            reward: Reward received during this macro.
            had_discovery: Whether any new discoveries were made.
        """
        if macro_idx not in self._per_macro_success:
            self._per_macro_success[macro_idx] = {"hits": 0.0, "total": 0.0, "reward_sum": 0.0}
        stats = self._per_macro_success[macro_idx]
        stats["total"] += 1.0
        stats["reward_sum"] += reward
        if had_discovery:
            stats["hits"] += 1.0

    def store_transition(
        self,
        state: torch.Tensor,
        macro: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ):
        """Store a macro-level transition in the replay buffer.
        
        Args:
            state: State when macro was selected.
            macro: MacroIntent index.
            reward: Option-level reward (accumulated over macro's duration).
            next_state: State after macro completed (or step ended).
            done: Whether episode ended.
        """
        self.replay_buffer.add(state, macro, reward, next_state, done)

    def update(self) -> Optional[float]:
        """Perform a Double DQN update if enough data is available.
        
        Returns:
            Loss value if update was performed, None otherwise.
        """
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return None

        if self.total_steps % self.config.update_freq != 0:
            return None

        # Sample batch
        states, macros, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size,
        )
        states = states.to(self.device)
        macros = macros.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # ── Double DQN: online selects, target evaluates ──────────────
        # Online network selects best next action
        with torch.no_grad():
            next_q_online = self.online_net(next_states)
            next_actions = next_q_online.argmax(dim=1)
            # Target network evaluates selected action
            next_q_target = self.target_net(next_states)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # Bellman target
            targets = rewards + self.config.gamma * next_q * (1 - dones)

        # Current Q-values
        current_q = self.online_net(states).gather(1, macros.unsqueeze(1)).squeeze(1)

        # TD error for metrics
        td_error = (targets - current_q).detach().abs().mean().item()

        # Huber loss (smooth L1) for stability
        loss = F.smooth_l1_loss(current_q, targets)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.config.grad_clip)
        self.optimizer.step()

        # Soft target update
        self._soft_update_target()
        self.update_count += 1

        # Record metrics
        loss_val = loss.item()
        self.training_metrics["loss"].append(loss_val)
        self.training_metrics["q_values"].append(current_q.mean().item())
        self.training_metrics["epsilon"].append(self.epsilon)
        self.training_metrics["td_errors"].append(td_error)

        return loss_val

    def _soft_update_target(self):
        """Polyak averaging: target ← τ * online + (1 - τ) * target."""
        tau = self.config.tau
        for target_param, online_param in zip(
            self.target_net.parameters(), self.online_net.parameters()
        ):
            target_param.data.copy_(
                tau * online_param.data + (1.0 - tau) * target_param.data
            )

    def reset_episode(self):
        """Reset per-episode tracking."""
        self._episode_macros = []
        self._last_macro = None
        # R57 Layer 1 resets
        self._macro_hold_counter = 0
        self._stagnation_counter = 0
        self._last_phase_name = None

    def get_macro_stats(self) -> Dict[str, Any]:
        """Get statistics about macro-intent selection this episode.
        
        Returns:
            Dict with macro distribution, switch count, etc.
        """
        if not self._episode_macros:
            return {"count": 0, "distribution": {}, "switches": 0}

        # Count distribution
        dist: Dict[str, int] = {}
        for m in self._episode_macros:
            name = MacroIntent(m).name
            dist[name] = dist.get(name, 0) + 1

        # Count switches (macro changed from previous step)
        switches = sum(
            1 for i in range(1, len(self._episode_macros))
            if self._episode_macros[i] != self._episode_macros[i - 1]
        )

        return {
            "count": len(self._episode_macros),
            "distribution": dist,
            "switches": switches,
            "unique_macros": len(dist),
            "dominant_macro": max(dist, key=dist.get) if dist else None,
        }

    def get_confidence_metrics(self) -> Dict[str, float]:
        """Get DDQN confidence metrics for mentor engagement decisions."""
        recent_td = self.training_metrics.get("td_errors", [])[-10:]
        recent_q = self.training_metrics.get("q_values", [])[-10:]

        return {
            "avg_td_error": sum(recent_td) / max(len(recent_td), 1) if recent_td else 0.0,
            "avg_q_value": sum(recent_q) / max(len(recent_q), 1) if recent_q else 0.0,
            "epsilon": self.epsilon,
            "buffer_size": len(self.replay_buffer),
            "update_count": self.update_count,
        }

    # ─── Persistence ──────────────────────────────────────────────────────

    def state_dict(self) -> Dict[str, Any]:
        """Get serializable state for checkpointing."""
        return {
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "update_count": self.update_count,
            "config": {
                "state_dim": self.config.state_dim,
                "num_macros": self.config.num_macros,
                "hidden_dims": self.config.hidden_dims,
                "learning_rate": self.config.learning_rate,
                "gamma": self.config.gamma,
            },
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Load from checkpoint."""
        self.online_net.load_state_dict(state["online_net"])
        self.target_net.load_state_dict(state["target_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.epsilon = state.get("epsilon", self.config.epsilon_start)
        self.total_steps = state.get("total_steps", 0)
        self.update_count = state.get("update_count", 0)
        logger.info(
            f"DDQNMacro loaded: steps={self.total_steps}, "
            f"updates={self.update_count}, ε={self.epsilon:.3f}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# R57 LAYER 1: STAGNATION OVERRIDE HELPER
# ═══════════════════════════════════════════════════════════════════════════════

# Phase → macro to force when stagnated (most forward-biased for that phase)
_STAGNATION_OVERRIDE_MAP: Dict[str, MacroIntent] = {
    "RECON": MacroIntent.SERVICE_ENUM,
    "ENUMERATION": MacroIntent.CREDENTIAL_CHAIN,
    "EXPLOITATION": MacroIntent.SERVICE_EXPLOIT,   # EP8 R56 fix — force exploitation
    "PRIVILEGE_ESCALATION": MacroIntent.PRIV_ESC,
    "LATERAL_MOVEMENT": MacroIntent.LATERAL,
    "POST_EXPLOITATION": MacroIntent.EXFIL,
    "EXFILTRATION": MacroIntent.EXFIL,
}


def _get_stagnation_override(
    phase_name: str,
    allowed: List[MacroIntent],
) -> Optional[MacroIntent]:
    """Get the forward-biased macro to force during stagnation.
    
    Returns the most appropriate macro for breaking out of the current
    phase, or None if no override is suitable.
    """
    override = _STAGNATION_OVERRIDE_MAP.get(phase_name)
    if override is not None and override in allowed:
        return override
    # Fallback: pick the last (most forward) allowed macro
    if allowed:
        return allowed[-1]
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# OPTION-LEVEL REWARD SHAPING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_macro_reward(
    macro: MacroIntent,
    step_reward: float,
    phase_name: str,
    discoveries: Dict[str, Any],
    prev_phase: Optional[str] = None,
    prev_macro: Optional[int] = None,
) -> float:
    """Compute option-level reward for a macro-intent.
    
    R57 Layer 1: Added macro switch penalty to discourage thrashing.
    
    Shapes reward to encourage:
    - Phase-appropriate macro selection
    - Forward phase progression
    - Discovery-rich exploration
    - Macro persistence (switch penalty)
    - Minimal looping
    
    Args:
        macro: The selected macro-intent.
        step_reward: Raw step reward from environment.
        phase_name: Current phase name.
        discoveries: New discoveries this step.
        prev_phase: Previous phase name (for transition detection).
        prev_macro: Previous macro index (for switch penalty). R57.
        
    Returns:
        Shaped macro-level reward.
    """
    reward = step_reward

    # ── Phase alignment bonus ──
    # Reward macro selections that match the current phase
    allowed = PHASE_MACRO_ALLOWLIST.get(phase_name, [])
    if macro in allowed:
        reward += 2.0  # Aligned with phase
    else:
        reward -= 3.0  # Phase-misaligned (shouldn't happen with masking)

    # ── R57: Macro switch penalty ──
    # Penalize frequent macro switching to encourage persistence
    if prev_macro is not None and prev_macro != macro.value:
        reward -= 2.0  # Switch cost — encourages macro stability

    # ── Forward progression bonus ──
    if prev_phase and prev_phase != phase_name:
        curr_ord = _get_phase_ordinal(phase_name)
        prev_ord = _get_phase_ordinal(prev_phase)
        if curr_ord > prev_ord:
            reward += 5.0 * (curr_ord - prev_ord)  # Forward progression
        elif curr_ord < prev_ord:
            reward -= 10.0  # Phase regression penalty

    # ── Discovery bonus at macro level ──
    num_discoveries = 0
    if isinstance(discoveries, dict):
        for key, val in discoveries.items():
            if isinstance(val, (set, list)):
                num_discoveries += len(val)
            elif isinstance(val, bool) and val:
                num_discoveries += 1
    if num_discoveries > 0:
        reward += 1.0 * num_discoveries

    return reward
