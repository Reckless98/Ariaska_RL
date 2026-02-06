#!/usr/bin/env python3
"""
core/algorithms/command_action_mapper.py — ARIASKA Command↔Action Mapper v1.0
🗺️ Maps PPO integer actions to concrete CommandTemplates per agent role.

Bridges the gap between PPO's discrete action space and the 144-command
registry by providing:
  - Per-role command pools (15-35 commands each, non-overlapping)
  - Precondition-based action masking (invalid → logit -∞)
  - Bidirectional mapping (action_idx ↔ CommandTemplate)
  - State-aware mask generation for safe PPO sampling

Each role gets a curated, non-overlapping subset of the registry so that
PPO networks remain tractable (action_dim ≈ 15-35 per role).

Usage:
    mapper = CommandActionMapper("RedAgent")
    mask = mapper.get_action_mask(state_flags)  # (action_dim,) bool tensor
    action_idx = ppo.select_action(state, mask)
    template = mapper.action_to_command(action_idx)
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any

import torch

from core.commands.command_registry import (
    COMMAND_REGISTRY,
    AttackPhase,
    CommandTemplate,
    get_valid_commands_for_state,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ROLE → COMMAND ASSIGNMENTS
# Comprehensive, non-overlapping assignment of every useful command to a role.
# ═══════════════════════════════════════════════════════════════════════════════

ROLE_COMMAND_ASSIGNMENTS: Dict[str, List[str]] = {
    # ─── Scout: Recon & Discovery ────────────────────────────────────────────
    "ScoutAgent": [
        # Network scanning
        "nmap_quick_scan", "nmap_top_ports", "nmap_full_tcp", "nmap_service_version",
        "nmap_vuln_scan", "nmap_udp_scan", "nmap_os_detection", "masscan_fast",
        "nmap_stealth_scan", "nmap_comprehensive",
        # DNS & OSINT
        "dig_any", "dig_axfr", "dnsrecon", "dns_enum", "whois_lookup",
        # Web fingerprinting
        "whatweb", "curl_headers",
        # SNMP/NFS discovery
        "snmpwalk", "onesixtyone", "showmount", "nfs_mount",
    ],

    # ─── Red: Offensive Exploitation ─────────────────────────────────────────
    "RedAgent": [
        # Brute force
        "hydra_ssh", "hydra_ftp", "hydra_smb", "hydra_http_form",
        # SQLi
        "sqlmap_get", "sqlmap_post", "sqlmap_test", "sqlmap_shell",
        # CrackMapExec
        "crackmapexec_smb_bruteforce", "crackmapexec_winrm", "cme_exec_command",
        "cme_smb_bruteforce",
        # Impacket exploitation
        "impacket_psexec", "impacket_smbexec", "impacket_wmiexec",
        "impacket_GetNPUsers", "impacket_GetUserSPNs",
        "impacket_atexec", "impacket_dcomexec",
        # Remote access
        "evil_winrm", "evil_winrm_hash", "ssh_login", "mysql_login", "mssql_login",
        # Vuln scanning
        "nikto_scan", "nuclei_scan", "searchsploit",
        # Metasploit
        "msfconsole_exploit", "msfvenom_payload", "msfconsole_auto",
        # Kerberos
        "rubeus_asreproast", "rubeus_kerberoast",
        # Privesc (offensive)
        "sudo_list", "find_suid", "find_capabilities", "find_writable_etc",
        "kernel_exploit_check", "sudo_check", "find_sgid",
        "winpeas", "whoami_all", "systeminfo", "windows_exploit_suggester",
        "accesschk_services", "powerup",
        # Post-exploit cred harvesting
        "impacket_secretsdump", "secretsdump_dc",
        "mimikatz_logonpasswords", "mimikatz_sam", "mimikatz_dcsync",
        "credential_dump", "hashdump",
        # Lateral (offensive)
        "impacket_pth_psexec", "crackmapexec_pth",
        "responder", "ntlmrelayx", "winrm_exec",
    ],

    # ─── Blue: Defensive Analysis ────────────────────────────────────────────
    "BlueAgent": [
        # NOTE: Blue uses custom_commands from AGENT_ROLES, not registry.
        # This list is for mapping purposes — actual commands are generated
        # by SmartCoach._get_blue_agent_command().
        # We create synthetic "blue_*" action indices for the 18 custom commands.
    ],

    # ─── Orion: Strategic Web/AD Scanning ────────────────────────────────────
    "OrionAgent": [
        # Directory brute force
        "gobuster_dir", "gobuster_vhost",
        "ffuf_fuzz", "wfuzz_params", "dirsearch", "feroxbuster",
        # LDAP/AD
        "ldapsearch_base", "ldapsearch_users", "windapsearch",
        "cme_ldap_users",
        # WordPress/CMS
        "wpscan",
        # Kerberos recon
        "certipy_find",
        # AD collection
        "bloodhound_python", "sharphound",
        # FTP anonymous
        "ftp_anonymous",
        # Redis
        "redis_cli",
        # Port knocking
        "knock", "nmap_port_knock",
    ],

    # ─── Shadow: Stealth & Persistence ───────────────────────────────────────
    "ShadowAgent": [
        # SMB/RPC enumeration (quiet)
        "smbclient_list", "smbclient_connect", "smbclient_auth",
        "smbmap_shares", "enum4linux_full", "enum4linux_ng", "enum4linux_scan",
        "rpcclient_null", "rpcclient_enumdomusers",
        # SSH enumeration
        "ssh_audit", "ssh_key_login",
        # Privesc recon (stealthy)
        "linpeas", "linpeas_local", "pspy", "pspy_monitor",
        "cron_check", "writable_etc_passwd", "capability_check",
        "docker_privesc", "lxd_privesc",
        # Tunneling & pivoting
        "chisel_server", "chisel_client", "ligolo_agent",
        "ssh_tunnel_local", "ssh_tunnel_dynamic", "ssh_lateral",
        "pivot_scan", "nmap_pivot", "proxychains_scan",
        # Persistence
        "cron_backdoor", "ssh_key_persistence",
        "keylogger_deploy", "history_dump", "network_config_dump", "ssh_key_harvest",
        # Exfiltration
        "nc_exfil", "curl_exfil", "scp_exfil", "exfil_data",
        "dns_exfil", "icmp_exfil", "base64_exfil", "smb_exfil",
        # Cleanup
        "cleanup_logs", "timestomp",
        # Impacket registry (stealthy)
        "impacket_reg",
        # CrackMapExec shares (stealthy)
        "cme_smb_shares",
        # Certipy
        "certipy_req",
    ],
}

# Blue agent's custom commands (not in the registry)
BLUE_CUSTOM_COMMANDS: List[Tuple[str, str]] = [
    ("netstat -tlnp", "List listening TCP ports"),
    ("ss -tlnp", "Socket statistics"),
    ("ps aux --sort=-%mem | head -20", "Top memory processes"),
    ("last -n 15", "Recent logins"),
    ("cat /var/log/auth.log 2>/dev/null | tail -30", "Auth logs"),
    ("who", "Logged in users"),
    ("w", "User activity"),
    ("lsof -i -P -n | head -30", "Network connections"),
    ("find /tmp -type f -mmin -30 2>/dev/null | head -20", "Recent /tmp files"),
    ("cat /etc/passwd | grep -v nologin", "Users with shells"),
    ("crontab -l 2>/dev/null", "Scheduled tasks"),
    ("systemctl list-units --type=service --state=running | head -20", "Running services"),
    ("df -h", "Disk usage"),
    ("free -m", "Memory usage"),
    ("uptime", "System uptime"),
    ("id", "Current user"),
    ("env | head -20", "Environment variables"),
    ("ls -la /home", "Home directories"),
]


class CommandActionMapper:
    """Maps PPO integer actions to concrete CommandTemplates for a given role.

    Provides bidirectional mapping and state-based action masking so that
    PPO can safely sample from only valid, role-appropriate commands.

    Args:
        role: Agent role name (e.g. "RedAgent", "ScoutAgent").

    Attributes:
        role: The agent role this mapper serves.
        action_dim: Number of discrete actions available.
        commands: Ordered list of (template_name, CommandTemplate | None).
        name_to_idx: Reverse mapping template_name → action index.
    """

    def __init__(self, role: str):
        self.role = role
        self.commands: List[Tuple[str, Optional[CommandTemplate]]] = []
        self.name_to_idx: Dict[str, int] = {}
        self._build_action_space()

    def _build_action_space(self):
        """Build the ordered list of commands for this role."""
        if self.role == "BlueAgent":
            # Blue uses custom commands, not registry
            for i, (cmd, desc) in enumerate(BLUE_CUSTOM_COMMANDS):
                synthetic_name = f"blue_cmd_{i}"
                self.commands.append((synthetic_name, None))
                self.name_to_idx[synthetic_name] = i
                # Also map the raw command string
                self.name_to_idx[cmd] = i
        else:
            assigned = ROLE_COMMAND_ASSIGNMENTS.get(self.role, [])
            for name in assigned:
                template = COMMAND_REGISTRY.get(name)
                if template:
                    idx = len(self.commands)
                    self.commands.append((name, template))
                    self.name_to_idx[name] = idx
                else:
                    logger.debug(f"CommandActionMapper({self.role}): "
                                 f"skipping unknown command '{name}'")

        self.action_dim = len(self.commands)
        logger.info(
            f"CommandActionMapper({self.role}): "
            f"{self.action_dim} actions mapped"
        )

    # ── Lookups ──────────────────────────────────────────────────────────────

    def action_to_command(self, action_idx: int) -> Optional[CommandTemplate]:
        """Convert a PPO action index to a CommandTemplate.

        Args:
            action_idx: Integer action from PPO.

        Returns:
            The corresponding CommandTemplate, or None for Blue custom cmds.
        """
        if 0 <= action_idx < self.action_dim:
            return self.commands[action_idx][1]
        return None

    def action_to_name(self, action_idx: int) -> Optional[str]:
        """Get the template name for an action index."""
        if 0 <= action_idx < self.action_dim:
            return self.commands[action_idx][0]
        return None

    def command_to_action(self, template_name: str) -> int:
        """Convert a template name to an action index.

        Returns:
            Action index, or -1 if not found.
        """
        return self.name_to_idx.get(template_name, -1)

    def get_blue_command(self, action_idx: int) -> Optional[Tuple[str, str]]:
        """Get the raw (command_string, description) for a Blue agent action.

        Returns:
            (command, description) tuple, or None if out of range.
        """
        if self.role != "BlueAgent":
            return None
        if 0 <= action_idx < len(BLUE_CUSTOM_COMMANDS):
            return BLUE_CUSTOM_COMMANDS[action_idx]
        return None

    # ── Action Masking ───────────────────────────────────────────────────────

    def get_action_mask(
        self,
        state_flags: Dict[str, Any],
        used_commands: Optional[Set[str]] = None,
        max_repeats: int = 3,
    ) -> torch.Tensor:
        """Compute a boolean action mask based on current state.

        A mask value of True means the action is VALID (can be selected).
        False means it should be masked out (logit set to -inf).

        Args:
            state_flags: Current environment state flags.
            used_commands: Set of command names already used this episode.
            max_repeats: Block commands used more than this many times.

        Returns:
            Boolean tensor of shape (action_dim,).
        """
        mask = torch.ones(self.action_dim, dtype=torch.bool)

        if self.role == "BlueAgent":
            # All Blue commands are always valid (no preconditions)
            return mask

        active_flags = set(k for k, v in state_flags.items() if v)
        used = used_commands or set()

        for idx, (name, template) in enumerate(self.commands):
            if template is None:
                continue

            # Check preconditions
            if template.preconditions and not template.preconditions.issubset(active_flags):
                mask[idx] = False
                continue

            # Check max repeats
            if used_commands is not None and name in used:
                # Count occurrences - we'll use a simple heuristic
                # The full count would need the command_history, but for masking
                # we just check if it's been used at all (anti-spam)
                pass  # Allow repeats up to max_repeats (tracked externally)

        # Ensure at least one action is valid (fallback safety)
        if not mask.any():
            # Enable all commands with no preconditions
            for idx, (name, template) in enumerate(self.commands):
                if template and not template.preconditions:
                    mask[idx] = True
            # If still nothing, enable everything
            if not mask.any():
                mask[:] = True

        return mask

    def get_action_mask_with_counts(
        self,
        state_flags: Dict[str, Any],
        command_counts: Dict[str, int],
        max_repeats: int = 4,
    ) -> torch.Tensor:
        """Compute action mask considering per-command repeat counts.

        More precise than get_action_mask: blocks commands used ≥ max_repeats.

        Args:
            state_flags: Current environment state flags.
            command_counts: Dict mapping template_name → times used this episode.
            max_repeats: Maximum allowed uses per command.

        Returns:
            Boolean tensor of shape (action_dim,).
        """
        mask = torch.ones(self.action_dim, dtype=torch.bool)

        if self.role == "BlueAgent":
            return mask

        active_flags = set(k for k, v in state_flags.items() if v)

        for idx, (name, template) in enumerate(self.commands):
            if template is None:
                continue

            # Precondition check
            if template.preconditions and not template.preconditions.issubset(active_flags):
                mask[idx] = False
                continue

            # Repeat count check
            if command_counts.get(name, 0) >= max_repeats:
                mask[idx] = False

        # Safety: ensure at least one valid action
        if not mask.any():
            for idx, (name, template) in enumerate(self.commands):
                if template and not template.preconditions:
                    mask[idx] = True
            if not mask.any():
                mask[:] = True

        return mask

    # ── Scoring / Ranking ────────────────────────────────────────────────────

    def get_discovery_scores(
        self,
        state_flags: Dict[str, Any],
        discoveries: Dict[str, Set[str]],
    ) -> torch.Tensor:
        """Score commands by discovery potential (higher = more promising).

        Uses CommandTemplate metadata (follows_after, enables, typical_reward)
        and current discoveries to rank commands by strategic value.

        Args:
            state_flags: Current state flags.
            discoveries: Current discoveries dict.

        Returns:
            Float tensor of shape (action_dim,) with scores ∈ [0, 1].
        """
        scores = torch.zeros(self.action_dim)

        if self.role == "BlueAgent":
            # Uniform scoring for Blue
            scores[:] = 1.0 / self.action_dim
            return scores

        active_flags = set(k for k, v in state_flags.items() if v)
        discovered_commands = set()
        if discoveries:
            for disc_type, values in discoveries.items():
                if isinstance(values, set):
                    discovered_commands.update(values)

        for idx, (name, template) in enumerate(self.commands):
            if template is None:
                continue

            score = 0.0

            # Base score from typical_reward
            score += min(template.typical_reward / 10.0, 1.0)

            # Bonus if follows_after commands have been executed
            if template.follows_after:
                matched = sum(1 for fa in template.follows_after
                              if fa in discovered_commands or fa in active_flags)
                score += 0.3 * (matched / len(template.follows_after))

            # Bonus if this command enables new things
            if template.enables:
                unblocked = sum(1 for e in template.enables
                                if e not in active_flags)
                score += 0.2 * (unblocked / len(template.enables))

            # Higher score for higher phases (progression)
            phase_bonus = {
                AttackPhase.RECON: 0.0,
                AttackPhase.ENUMERATION: 0.1,
                AttackPhase.EXPLOITATION: 0.3,
                AttackPhase.PRIVILEGE_ESCALATION: 0.4,
                AttackPhase.LATERAL_MOVEMENT: 0.5,
                AttackPhase.EXFILTRATION: 0.6,
                AttackPhase.POST_EXPLOITATION: 0.5,
            }
            score += phase_bonus.get(template.phase, 0.0)

            scores[idx] = score

        # Normalize to [0, 1]
        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score

        return scores

    # ── Utilities ────────────────────────────────────────────────────────────

    def get_all_command_names(self) -> List[str]:
        """Get ordered list of all command names in this mapper."""
        return [name for name, _ in self.commands]

    def __repr__(self) -> str:
        return f"CommandActionMapper(role={self.role}, actions={self.action_dim})"


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY / CACHE
# ═══════════════════════════════════════════════════════════════════════════════

_mapper_cache: Dict[str, CommandActionMapper] = {}


def get_mapper(role: str) -> CommandActionMapper:
    """Get or create a CommandActionMapper for the given role.

    Cached so that the same mapper is reused across episodes.

    Args:
        role: Agent role name.

    Returns:
        CommandActionMapper for that role.
    """
    if role not in _mapper_cache:
        _mapper_cache[role] = CommandActionMapper(role)
    return _mapper_cache[role]


def get_all_mappers() -> Dict[str, CommandActionMapper]:
    """Get mappers for all 5 agent roles."""
    roles = ["ScoutAgent", "RedAgent", "BlueAgent", "OrionAgent", "ShadowAgent"]
    return {role: get_mapper(role) for role in roles}


def validate_assignments() -> Dict[str, Any]:
    """Validate that all registry commands are assigned to exactly one role.

    Returns:
        Dict with stats and any unassigned/duplicate commands.
    """
    all_assigned: Dict[str, str] = {}  # command → role
    duplicates: List[Tuple[str, str, str]] = []
    unassigned: List[str] = []

    for role, commands in ROLE_COMMAND_ASSIGNMENTS.items():
        for cmd_name in commands:
            if cmd_name in all_assigned:
                duplicates.append((cmd_name, all_assigned[cmd_name], role))
            else:
                all_assigned[cmd_name] = role

    for cmd_name in COMMAND_REGISTRY:
        if cmd_name not in all_assigned:
            unassigned.append(cmd_name)

    return {
        "total_assigned": len(all_assigned),
        "total_registry": len(COMMAND_REGISTRY),
        "unassigned": unassigned,
        "duplicates": duplicates,
        "per_role": {role: len(cmds) for role, cmds in ROLE_COMMAND_ASSIGNMENTS.items()},
    }
