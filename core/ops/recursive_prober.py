"""core/ops/recursive_prober.py — Phase 41: Recursive web enumeration.

When web paths are discovered, automatically queues deeper probing
at those paths using tools like gobuster/feroxbuster.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("ariaska.ops.recursive_prober")


@dataclass
class RecursiveProberConfig:
    """Configuration for recursive web probing."""
    enabled: bool = True
    max_depth: int = 3
    interesting_paths: List[str] = field(
        default_factory=lambda: [
            "/api", "/admin", "/login", "/upload", "/config",
            "/backup", "/.git", "/dev", "/debug", "/internal",
            "/v1", "/v2", "/graphql", "/swagger", "/docs",
        ]
    )
    tools: List[str] = field(
        default_factory=lambda: ["gobuster", "feroxbuster", "dirb"]
    )
    max_queued: int = 10
    wordlist: str = "/usr/share/wordlists/dirb/common.txt"
    path_priority: Dict[str, int] = field(
        default_factory=lambda: {
            "/api": 1, "/admin": 2, "/backup": 3,
            "/.git": 4, "/config": 5, "/upload": 6,
        }
    )


class RecursiveProber:
    """Generates recursive web enumeration probes from discovered paths."""

    def __init__(self, config: Optional[RecursiveProberConfig] = None) -> None:
        self.config = config or RecursiveProberConfig()
        self._queue: Deque[Tuple[str, str, int]] = deque()  # (base_url, path, depth)
        self._probed: Set[str] = set()
        self._discoveries: Dict[str, List[str]] = {}

    def feed_discoveries(
        self, paths_found: List[str], base_url: str
    ) -> None:
        """Feed newly discovered web paths for potential recursive probing.

        Args:
            paths_found: List of discovered paths (e.g. ["/api", "/login"]).
            base_url: Base URL (e.g. "http://10.10.10.1:80").
        """
        if not self.config.enabled:
            return

        for path in paths_found:
            normalized = "/" + path.strip("/")
            full_key = f"{base_url}{normalized}"

            if full_key in self._probed:
                continue

            if len(self._queue) >= self.config.max_queued:
                break

            # Check if interesting
            is_interesting = any(
                normalized.startswith(ip) or normalized == ip
                for ip in self.config.interesting_paths
            )
            if is_interesting:
                self._queue.append((base_url, normalized, 1))
                logger.debug("Queued probe: %s%s (depth 1)", base_url, normalized)

    def get_next_probes(self, max_probes: int = 3) -> List[str]:
        """Get the next batch of probe commands.

        Args:
            max_probes: Maximum number of probes to return.

        Returns:
            List of command strings for web enumeration.
        """
        if not self.config.enabled or not self._queue:
            return []

        commands: List[str] = []
        to_remove: List[int] = []

        sorted_queue = sorted(
            self._queue,
            key=lambda x: self.config.path_priority.get(x[1], 99),
        )

        for base_url, path, depth in sorted_queue:
            if len(commands) >= max_probes:
                break

            full_key = f"{base_url}{path}"
            if full_key in self._probed or depth > self.config.max_depth:
                continue

            self._probed.add(full_key)
            tool = self.config.tools[0] if self.config.tools else "gobuster"
            cmd = self._build_probe_command(tool, base_url, path)
            commands.append(cmd)

        # Remove probed items from queue
        self._queue = deque(
            (b, p, d) for b, p, d in self._queue
            if f"{b}{p}" not in self._probed
        )
        return commands

    def record_result(
        self, path: str, found_subpaths: List[str], base_url: str = ""
    ) -> None:
        """Record results from a recursive probe and queue deeper probes.

        Args:
            path: The path that was probed.
            found_subpaths: Subpaths discovered under that path.
            base_url: Base URL for constructing deeper probes.
        """
        self._discoveries[path] = found_subpaths

        if not base_url:
            return

        for sp in found_subpaths:
            full_path = f"{path.rstrip('/')}/{sp.strip('/')}"
            full_key = f"{base_url}{full_path}"
            # Compute depth
            depth = full_path.count("/")
            if (
                full_key not in self._probed
                and depth <= self.config.max_depth
                and len(self._queue) < self.config.max_queued
            ):
                self._queue.append((base_url, full_path, depth))

    @property
    def queued_count(self) -> int:
        """Number of probes in queue."""
        return len(self._queue)

    @property
    def probed_count(self) -> int:
        """Number of paths already probed."""
        return len(self._probed)

    def _build_probe_command(
        self, tool: str, base_url: str, path: str
    ) -> str:
        """Build a probe command string."""
        url = f"{base_url}{path}"
        if tool == "gobuster":
            return f"gobuster dir -u {url} -w {self.config.wordlist} -q -t 20"
        elif tool == "feroxbuster":
            return f"feroxbuster -u {url} -w {self.config.wordlist} -q -t 20"
        elif tool == "dirb":
            return f"dirb {url} {self.config.wordlist} -S -r"
        return f"gobuster dir -u {url} -w {self.config.wordlist} -q"
