"""
core/ops/hosts_manager.py — Dynamic /etc/hosts management

Manages domain ↔ IP mappings for HTB and authorized targets.
Uses SudoHandler for privileged writes.
Rules:
  - Avoid duplicates.
  - Auto-generate common subdomains (ftp., dev., www.).
  - Thread-safe.
  - Respect dry-run mode.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import List, Optional, Set

logger = logging.getLogger("ariaska.ops.hosts")

# Standard subdomains auto-appended when a domain is registered
_DEFAULT_SUBDOMAINS = ("ftp", "dev", "www")

# Regex to validate hostname-ish strings
_DOMAIN_RE = re.compile(r"^[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?)*$")
_IP_RE = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")


class HostsManager:
    """
    Manages /etc/hosts entries for HTB / authorised target domains.

    Usage:
        from core.ops.sudo_handler import SudoHandler
        sudo = SudoHandler()
        hm = HostsManager(sudo)
        hm.ensure_entry("10.10.11.50", "soulmate.htb")
    """

    def __init__(self, sudo_handler: object, hosts_path: str = "/etc/hosts") -> None:
        """
        Args:
            sudo_handler: SudoHandler instance for privileged writes.
            hosts_path: Path to hosts file (overridable for testing).
        """
        self._sudo = sudo_handler
        self._hosts_path = hosts_path
        self._lock = threading.Lock()
        self._managed_entries: Set[str] = set()  # Track entries we've added
        logger.debug("HostsManager initialised (hosts=%s)", hosts_path)

    # ── Public API ────────────────────────────────────────────────────────

    def ensure_entry(
        self,
        ip: str,
        domain: str,
        subdomains: Optional[List[str]] = None,
    ) -> bool:
        """
        Ensure /etc/hosts contains the given IP → domain mapping.

        Auto-generates ftp.<domain>, dev.<domain>, www.<domain> subdomains
        plus any extra subdomains provided.

        Args:
            ip: Target IP address.
            domain: Primary domain (e.g. "soulmate.htb").
            subdomains: Additional subdomains to include.

        Returns:
            True if entry was added or already present, False on failure.
        """
        with self._lock:
            return self._ensure_locked(ip, domain, subdomains)

    def add_vhost(self, ip: str, vhost: str) -> bool:
        """
        Add a single discovered vhost to /etc/hosts.

        Args:
            ip: Target IP address.
            vhost: Virtual hostname to add.

        Returns:
            True if added or already present.
        """
        if not self._validate_domain(vhost):
            logger.warning("Invalid vhost rejected: %s", vhost)
            return False
        with self._lock:
            if self.has_entry(ip, vhost):
                logger.debug("vhost already in hosts: %s → %s", ip, vhost)
                return True
            return self._append_line(f"{ip}\t{vhost}")

    def has_entry(self, ip: str, domain: str) -> bool:
        """
        Check if /etc/hosts already contains ip → domain mapping.

        Thread-safe read (no lock needed — file reads are atomic enough).
        """
        try:
            content = self._read_hosts()
        except OSError:
            return False
        # Normalise
        ip_stripped = ip.strip()
        domain_lower = domain.strip().lower()
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0] == ip_stripped:
                if domain_lower in [p.lower() for p in parts[1:]]:
                    return True
        return False

    @property
    def managed_entries(self) -> Set[str]:
        """Return set of domain entries managed by this session."""
        return set(self._managed_entries)

    # ── Private ───────────────────────────────────────────────────────────

    def _ensure_locked(self, ip: str, domain: str, subdomains: Optional[List[str]]) -> bool:
        """Core ensure logic — call with lock held."""
        if not _IP_RE.match(ip):
            logger.warning("Invalid IP rejected: %s", ip)
            return False
        if not self._validate_domain(domain):
            logger.warning("Invalid domain rejected: %s", domain)
            return False

        # Build full hostname list
        hostnames = [domain]
        for sub in _DEFAULT_SUBDOMAINS:
            sub_host = f"{sub}.{domain}"
            if sub_host not in hostnames:
                hostnames.append(sub_host)
        if subdomains:
            for sub in subdomains:
                if self._validate_domain(sub) and sub not in hostnames:
                    hostnames.append(sub)

        # Filter out already-present entries
        missing = [h for h in hostnames if not self.has_entry(ip, h)]
        if not missing:
            logger.debug("All entries already present for %s → %s", ip, domain)
            return True

        # Build line
        line = f"{ip}\t{' '.join(missing)}"
        success = self._append_line(line)
        if success:
            for h in missing:
                self._managed_entries.add(h)
        return success

    def _append_line(self, line: str) -> bool:
        """Append a line to /etc/hosts via SudoHandler."""
        # Use tee -a to append
        cmd = f"tee -a {self._hosts_path} <<< '{line}'"
        result = self._sudo.execute_privileged(cmd)
        if result.success or result.dry_run:
            logger.info("hosts entry added: %s", line)
            return True
        logger.warning("Failed to add hosts entry: %s — %s", line, result.stderr[:120])
        return False

    def _read_hosts(self) -> str:
        """Read the hosts file contents."""
        try:
            with open(self._hosts_path, "r", encoding="utf-8") as fh:
                return fh.read()
        except OSError as exc:
            logger.warning("Cannot read %s: %s", self._hosts_path, exc)
            return ""

    @staticmethod
    def _validate_domain(domain: str) -> bool:
        """Validate domain string."""
        if not domain or len(domain) > 253:
            return False
        return bool(_DOMAIN_RE.match(domain))
