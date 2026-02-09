"""Hard Mode difficulty presets for Ariaska_RL.

Phase 6.6: Difficulty presets that block trivial exploitation paths
to force the agent to learn multi-step attack chains.

Presets:
    normal:  No restrictions — all services and commands available.
    medium:  Block instant-root shortcuts (ingreslock 1524, rsh/rlogin).
             Agent must use credential-based or exploit-based chains.
    hard:    Block all single-step root paths. Agent must chain multiple
             vulnerabilities (e.g., web app → credential → privesc → root).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Set

logger = logging.getLogger("ariaska.difficulty")


@dataclass(frozen=True)
class DifficultyPreset:
    """Immutable difficulty configuration."""

    name: str
    description: str
    # Command template names that are BLOCKED (will not be proposed by SmartCoach)
    blocked_commands: FrozenSet[str] = field(default_factory=frozenset)
    # Port numbers that simulated output should NOT show as "open"
    blocked_ports: FrozenSet[int] = field(default_factory=frozenset)
    # Services that are hidden from simulated scan output
    blocked_services: FrozenSet[str] = field(default_factory=frozenset)
    # Multiplier on required phase_progress thresholds (1.0 = normal)
    phase_threshold_multiplier: float = 1.0
    # Extra description appended to mentor system prompt
    mentor_hint: str = ""


# ═══════════════════════════════════════════════════════════════════════
# PRESET DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

NORMAL = DifficultyPreset(
    name="normal",
    description="No restrictions — all services available. Good for initial learning.",
    mentor_hint="",
)

MEDIUM = DifficultyPreset(
    name="medium",
    description="Block instant-root shortcuts (ingreslock, r-services). Forces credential or exploit chains.",
    blocked_commands=frozenset({
        "telnet_1524",          # Ingreslock instant root
        "nc_1524",              # Netcat to ingreslock
        "rsh_root",             # R-services unauthenticated root
        "rlogin_root",          # R-login unauthenticated root
        "rexec_root",           # R-exec unauthenticated root
        "nmap_port_1524",       # Don't even scan for ingreslock
    }),
    blocked_ports=frozenset({1524, 512, 513, 514}),
    blocked_services=frozenset({"ingreslock", "rexec", "rlogin", "rsh"}),
    phase_threshold_multiplier=1.0,
    mentor_hint=(
        "DIFFICULTY: MEDIUM — Ingreslock (port 1524) and r-services (512-514) are "
        "BLOCKED. Do NOT suggest telnet to 1524 or rsh/rlogin. Focus on: "
        "vsftpd 2.3.4 backdoor (port 21), Samba usermap_script (port 445), "
        "SSH brute-force, web application attacks (port 80), or database exploits."
    ),
)

HARD = DifficultyPreset(
    name="hard",
    description="Block ALL single-step root paths. Forces multi-step chains (web→creds→privesc).",
    blocked_commands=frozenset({
        # Ingreslock
        "telnet_1524", "nc_1524", "nmap_port_1524",
        # R-services
        "rsh_root", "rlogin_root", "rexec_root", "nmap_rservices",
        # Direct backdoor exploits
        "vsftpd_exploit", "vsftpd_backdoor",
        "unrealircd_exploit", "unrealircd_backdoor",
        "samba_exploit", "samba_usermap_script",
        # Java RMI direct exploit
        "java_rmi_exploit",
    }),
    blocked_ports=frozenset({1524, 512, 513, 514, 6667}),
    blocked_services=frozenset({
        "ingreslock", "rexec", "rlogin", "rsh", "unrealircd",
    }),
    phase_threshold_multiplier=1.5,
    mentor_hint=(
        "DIFFICULTY: HARD — All single-step root exploits are BLOCKED: "
        "ingreslock, r-services, vsftpd backdoor, UnrealIRCd backdoor, "
        "Samba usermap_script, Java RMI. You MUST use multi-step chains: "
        "1) Enumerate web apps on port 80/8180 "
        "2) Find credentials via SQL injection, brute-force, or NFS mount "
        "3) Use found creds for SSH/Telnet login "
        "4) Escalate privileges via kernel exploit or SUID binaries. "
        "Focus on: DVWA (port 80), Tomcat (8180), MySQL (3306), "
        "PostgreSQL (5432), NFS (2049), VNC (5900)."
    ),
)

# ═══════════════════════════════════════════════════════════════════════
# REGISTRY
# ═══════════════════════════════════════════════════════════════════════

DIFFICULTY_PRESETS: Dict[str, DifficultyPreset] = {
    "normal": NORMAL,
    "medium": MEDIUM,
    "hard": HARD,
}


def get_preset(name: str) -> DifficultyPreset:
    """Get a difficulty preset by name.

    Args:
        name: One of 'normal', 'medium', 'hard'.

    Returns:
        The corresponding DifficultyPreset.

    Raises:
        ValueError: If the preset name is unknown.
    """
    preset = DIFFICULTY_PRESETS.get(name.lower())
    if preset is None:
        valid = ", ".join(DIFFICULTY_PRESETS.keys())
        raise ValueError(f"Unknown difficulty '{name}'. Valid: {valid}")
    logger.info(f"[DIFFICULTY] Loaded preset: {preset.name} — {preset.description}")
    return preset


def is_command_blocked(command_name: str, preset: DifficultyPreset) -> bool:
    """Check if a command template name is blocked by the current difficulty.

    Args:
        command_name: The template name (e.g. 'telnet_1524').
        preset: Active difficulty preset.

    Returns:
        True if the command should be blocked.
    """
    return command_name in preset.blocked_commands


def filter_blocked_ports(ports: List[int], preset: DifficultyPreset) -> List[int]:
    """Remove blocked ports from a list (for simulated scan output).

    Args:
        ports: List of port numbers.
        preset: Active difficulty preset.

    Returns:
        Filtered list with blocked ports removed.
    """
    if not preset.blocked_ports:
        return ports
    return [p for p in ports if p not in preset.blocked_ports]
