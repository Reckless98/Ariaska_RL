"""
Scan exposure randomization for Ariaska training generalization.

R66: Prevents policy overfitting to a fixed initial scan/ports ordering.
Under seed control, randomizes:
  - Top-ports subset for early quick scans (20, 50, 100 ports)
  - Service probe intensity (light vs normal)
  - Ordering of standard first-action commands
  - Partial scan mode (sometimes scans fewer ports to force inference)

Usage:
    from core.analytics.scan_randomizer import ScanRandomizer
    sr = ScanRandomizer(seed=66, env_name="ms2")
    first_cmds = sr.get_randomized_initial_commands(target_ip)
    nmap_ports = sr.get_scan_ports()
    intensity = sr.get_probe_intensity()
"""
from __future__ import annotations

import logging
import random
from typing import List, Optional

logger = logging.getLogger("ariaska.analytics.scan_randomizer")

# Standard port sets for randomization
PORT_SETS = {
    "quick20": "21,22,23,25,80,110,111,139,443,445,993,995,1099,1433,1524,2049,3306,5432,5900,8080",
    "standard50": "21,22,23,25,53,80,110,111,135,139,143,443,445,993,995,1099,1433,1524,2049,3306,3389,5432,5900,5985,6667,8080,8180,8443,9090,9100",
    "wide100": "1-1024,1099,1433,1524,2049,3306,3389,5432,5900,5985,6667,8080,8180,8443,9090",
}

PROBE_INTENSITIES = {
    "light": "-sV --version-intensity 2",
    "normal": "-sV --version-intensity 5",
    "aggressive": "-sV --version-intensity 7 -A",
}

# Different first-action orderings for initial recon
INITIAL_ACTION_SETS = [
    # Set A: nmap first, then enum
    [
        "nmap -sV {ports} {target}",
        "nmap -sC {ports} {target}",
        "enum4linux -a {target}",
    ],
    # Set B: stealth scan first
    [
        "nmap -sS {ports} {target}",
        "nmap --script vuln {ports} {target}",
        "smbclient -L //{target} -N",
    ],
    # Set C: service-focused
    [
        "nmap {intensity} {ports} {target}",
        "nmap --script smb-vuln* -p 139,445 {target}",
        "rpcclient -U '' -N {target} -c 'srvinfo'",
    ],
    # Set D: wide then deep
    [
        "nmap -sS -p- --min-rate 5000 {target}",
        "nmap -sC -sV -p 21,22,139,445,3306 {target}",
        "showmount -e {target}",
    ],
]


class ScanRandomizer:
    """Controlled randomization of scan exposure for training generalization.
    
    Seeded for reproducibility. Each episode can call ``next_episode()``
    to advance the randomization while maintaining determinism across seeds.
    """

    def __init__(self, seed: int = 42, env_name: str = "ms2"):
        self.seed = seed
        self.env_name = env_name
        self._rng = random.Random(seed)
        self._episode = 0
        self._port_set_key: str = "standard50"
        self._intensity_key: str = "normal"
        self._action_set_idx: int = 0
        self._partial_mode: bool = False
        self._advance()  # Initialize for episode 0
        logger.info(
            f"ScanRandomizer: seed={seed}, env={env_name}"
        )

    def next_episode(self) -> None:
        """Advance to next episode's randomization."""
        self._episode += 1
        self._advance()

    def _advance(self) -> None:
        """Randomize parameters for current episode."""
        # Choose port set: bias toward standard, occasionally quick or wide
        r = self._rng.random()
        if r < 0.3:
            self._port_set_key = "quick20"
        elif r < 0.8:
            self._port_set_key = "standard50"
        else:
            self._port_set_key = "wide100"

        # Probe intensity
        r = self._rng.random()
        if r < 0.4:
            self._intensity_key = "light"
        elif r < 0.85:
            self._intensity_key = "normal"
        else:
            self._intensity_key = "aggressive"

        # Initial action set
        self._action_set_idx = self._rng.randint(0, len(INITIAL_ACTION_SETS) - 1)

        # Partial scan mode (20% of the time)
        self._partial_mode = self._rng.random() < 0.20

        logger.debug(
            f"ScanRandomizer EP{self._episode}: ports={self._port_set_key}, "
            f"intensity={self._intensity_key}, actions=set{self._action_set_idx}, "
            f"partial={self._partial_mode}"
        )

    def get_scan_ports(self) -> str:
        """Get port specification string for nmap."""
        ports = PORT_SETS[self._port_set_key]
        if self._partial_mode:
            # In partial mode, only scan a random subset of ports
            port_list = [p.strip() for p in ports.split(",") if "-" not in p]
            if len(port_list) > 5:
                subset = self._rng.sample(port_list, k=max(5, len(port_list) // 2))
                return ",".join(sorted(subset, key=lambda x: int(x) if x.isdigit() else 0))
        return ports

    def get_probe_intensity(self) -> str:
        """Get nmap probe intensity flags."""
        return PROBE_INTENSITIES[self._intensity_key]

    def get_randomized_initial_commands(self, target_ip: str) -> List[str]:
        """Get initial recon commands in randomized order.
        
        Returns formatted commands with target_ip and scan parameters filled in.
        """
        action_templates = INITIAL_ACTION_SETS[self._action_set_idx]
        ports_str = f"-p {self.get_scan_ports()}"
        intensity = self.get_probe_intensity()

        commands = []
        for tmpl in action_templates:
            cmd = tmpl.format(
                target=target_ip,
                ports=ports_str,
                intensity=intensity,
            )
            commands.append(cmd)

        # Shuffle the command order slightly (swap adjacent pairs sometimes)
        if self._rng.random() < 0.3 and len(commands) >= 2:
            i = self._rng.randint(0, len(commands) - 2)
            commands[i], commands[i + 1] = commands[i + 1], commands[i]

        return commands

    @property
    def current_config(self) -> dict:
        """Current randomization config for logging."""
        return {
            "episode": self._episode,
            "port_set": self._port_set_key,
            "intensity": self._intensity_key,
            "action_set": self._action_set_idx,
            "partial_mode": self._partial_mode,
        }
