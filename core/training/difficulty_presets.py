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

# ── MS3-specific difficulty presets ──────────────────────────────────

MS3_MEDIUM = DifficultyPreset(
    name="ms3_medium",
    description="MS3: Block instant-root vectors (UnrealIRCd backdoor, SSH vagrant sudo). Forces credential or exploit chains.",
    blocked_commands=frozenset({
        # UnrealIRCd backdoor (instant root, same vector as MS2 ingreslock)
        "unrealircd_exploit", "unrealircd_backdoor",
        # SSH vagrant → sudo su (trivial root)
        "ssh_login",  # Block default SSH with vagrant:vagrant direct path
    }),
    blocked_ports=frozenset({6667}),  # Hide UnrealIRCd
    blocked_services=frozenset({"unrealircd"}),
    phase_threshold_multiplier=1.2,
    mentor_hint=(
        "DIFFICULTY: MS3 MEDIUM — UnrealIRCd backdoor (port 6667) is BLOCKED. "
        "SSH with vagrant:vagrant sudo is RESTRICTED. Focus on: "
        "ProFTPD mod_copy (port 21, CVE-2015-3306) to write files then trigger RCE, "
        "Jenkins Groovy console (port 8484) if available, "
        "MySQL root no-password (port 3306) → UDF plugin for command exec, "
        "Tomcat (port 8080) → manager WAR deploy, "
        "WordPress admin:admin → plugin upload shell. "
        "You MUST use service-specific exploits, not default credentials."
    ),
)

MS3_HARD = DifficultyPreset(
    name="ms3_hard",
    description="MS3 Hard: Block all single-step root paths. Forces multi-step chains (service exploit → creds → privesc).",
    blocked_commands=frozenset({
        # All instant-root vectors
        "unrealircd_exploit", "unrealircd_backdoor",
        # Direct SSH/FTP with known creds
        "ssh_login",
        "hydra_ssh",  # Block SSH brute-force (vagrant:vagrant too easy)
        # Direct ProFTPD mod_copy (single-step file write)
        "ftp_anonymous",
        # Jenkins direct if scripting console open
        # Block simulated instant-access commands
        "telnet_1524", "nc_1524",
        "rsh_root", "rlogin_root", "rexec_root",
    }),
    blocked_ports=frozenset({6667, 1524, 512, 513, 514}),
    blocked_services=frozenset({"unrealircd", "ingreslock", "rexec", "rlogin", "rsh"}),
    phase_threshold_multiplier=1.5,
    mentor_hint=(
        "DIFFICULTY: MS3 HARD — ALL single-step root paths are BLOCKED: "
        "UnrealIRCd backdoor, SSH vagrant:vagrant, direct FTP anonymous, r-services. "
        "You MUST use MULTI-STEP attack chains:\n"
        "  1) Enumerate web services (Apache/Tomcat/WordPress/Jenkins)\n"
        "  2) Find credentials via MySQL (root no-pass), web app exploits, or NFS\n"
        "  3) Use found creds for service login\n"
        "  4) Escalate via kernel exploit, SUID binary, or sudo misconfiguration\n"
        "BEST CHAINS: MySQL UDF → command exec → shell, "
        "Tomcat manager → WAR deploy → www-data → kernel privesc, "
        "WordPress → plugin upload → shell → SUID find, "
        "ProFTPD mod_copy → write cron job → root, "
        "Samba enumeration → credential discovery → lateral movement."
    ),
)

DIFFICULTY_PRESETS: Dict[str, DifficultyPreset] = {
    "normal": NORMAL,
    "medium": MEDIUM,
    "hard": HARD,
    "ms3_medium": MS3_MEDIUM,
    "ms3_hard": MS3_HARD,
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
