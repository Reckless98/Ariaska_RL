"""
core/ops/domain_manager.py — Phase 38.2: Domain & VHost Tracking

Manages discovered domains, subdomains, and vhosts throughout
a session.  Provides a single source of truth for all
hostname/domain data gathered during an engagement.

Features:
  - Track primary domain and discovered subdomains
  - Auto-expand common subdomains (www, ftp, dev, api, etc.)
  - Deduplicated domain registry
  - Provides domain context for /etc/hosts management
  - Session-persistent domain state
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set

logger = logging.getLogger("ariaska.ops.domain_manager")

# ── Defaults ─────────────────────────────────────────────────────────────────

_COMMON_SUBDOMAINS: FrozenSet[str] = frozenset({
    "www", "ftp", "dev", "api", "admin", "mail",
    "webmail", "portal", "intranet", "staging",
})

_DOMAIN_REGEX = re.compile(
    r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$"
)

_SUBDOMAIN_REGEX = re.compile(
    r"(?:^|[\s/])([a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?"
    r"(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*\.[a-zA-Z]{2,})"
)


@dataclass
class DomainEntry:
    """A single tracked domain."""
    hostname: str
    source: str = "manual"  # manual, scan, vhost_enum, dns, bruteforce
    ip_address: str = ""
    is_primary: bool = False
    is_vhost: bool = False


class DomainManager:
    """
    Central domain and vhost tracker for the engagement.

    Usage:
        dm = DomainManager()
        dm.set_primary("permx.htb", ip="10.10.11.23")
        dm.add_domain("lms.permx.htb", source="vhost_enum")
        all_domains = dm.get_all_domains()
        hosts_entries = dm.get_hosts_entries()
    """

    def __init__(self) -> None:
        self._domains: Dict[str, DomainEntry] = {}
        self._primary_domain: Optional[str] = None
        self._primary_ip: str = ""
        logger.debug("DomainManager initialised")

    def set_primary(self, domain: str, ip: str = "") -> bool:
        """
        Set the primary engagement domain.

        Args:
            domain: The primary domain (e.g., 'permx.htb').
            ip: The target IP address.

        Returns:
            True if set successfully.
        """
        domain = domain.strip().lower()
        if not domain:
            return False

        self._primary_domain = domain
        self._primary_ip = ip

        entry = DomainEntry(
            hostname=domain,
            source="primary",
            ip_address=ip,
            is_primary=True,
        )
        self._domains[domain] = entry

        # Auto-expand common subdomains
        for sub in _COMMON_SUBDOMAINS:
            candidate = f"{sub}.{domain}"
            if candidate not in self._domains:
                self._domains[candidate] = DomainEntry(
                    hostname=candidate,
                    source="auto_expand",
                    ip_address=ip,
                    is_vhost=True,
                )

        logger.info("Primary domain set: %s (IP: %s)", domain, ip or "unknown")
        return True

    def add_domain(
        self,
        domain: str,
        source: str = "scan",
        ip: str = "",
        is_vhost: bool = False,
    ) -> bool:
        """
        Add a discovered domain/subdomain.

        Args:
            domain: The domain to add.
            source: How it was discovered.
            ip: Associated IP address.
            is_vhost: Whether this is a virtual host.

        Returns:
            True if newly added, False if already known.
        """
        domain = domain.strip().lower()
        if not domain:
            return False

        if domain in self._domains:
            # Update IP if not set
            if ip and not self._domains[domain].ip_address:
                self._domains[domain].ip_address = ip
            return False

        self._domains[domain] = DomainEntry(
            hostname=domain,
            source=source,
            ip_address=ip or self._primary_ip,
            is_vhost=is_vhost,
        )
        logger.info("Domain added: %s (source=%s)", domain, source)
        return True

    def extract_domains_from_output(self, output: str) -> List[str]:
        """
        Extract domain names from tool output.

        Args:
            output: Raw command output.

        Returns:
            List of newly discovered domain names.
        """
        if not output or not self._primary_domain:
            return []

        new_domains: List[str] = []
        primary = self._primary_domain

        for match in _SUBDOMAIN_REGEX.finditer(output):
            candidate = match.group(1).lower()
            # Only accept subdomains of our primary domain
            if candidate.endswith(f".{primary}") and candidate != primary:
                if self.add_domain(candidate, source="extraction", is_vhost=True):
                    new_domains.append(candidate)

        return new_domains

    def get_all_domains(self) -> List[str]:
        """Return all tracked domain names."""
        return sorted(self._domains.keys())

    def get_confirmed_domains(self) -> List[str]:
        """Return domains that are not auto-expanded (confirmed real)."""
        return sorted(
            name for name, entry in self._domains.items()
            if entry.source != "auto_expand"
        )

    def get_primary_domain(self) -> Optional[str]:
        """Return the primary engagement domain."""
        return self._primary_domain

    def get_primary_ip(self) -> str:
        """Return the primary target IP."""
        return self._primary_ip

    def get_hosts_entries(self) -> List[Dict[str, str]]:
        """
        Return entries suitable for /etc/hosts.

        Returns:
            List of dicts with 'ip' and 'hostname' keys.
        """
        entries: List[Dict[str, str]] = []
        for name, entry in sorted(self._domains.items()):
            ip = entry.ip_address or self._primary_ip
            if ip:
                entries.append({"ip": ip, "hostname": name})
        return entries

    def get_vhosts(self) -> List[str]:
        """Return all virtual hosts."""
        return sorted(
            name for name, entry in self._domains.items()
            if entry.is_vhost
        )

    def has_domain(self, domain: str) -> bool:
        """Check if a domain is tracked."""
        return domain.strip().lower() in self._domains

    def domain_count(self) -> int:
        """Return total number of tracked domains."""
        return len(self._domains)

    def get_context(self) -> Dict[str, Any]:
        """
        Return domain context for LLM prompts or state encoding.

        Returns:
            Dict with primary_domain, primary_ip, domain_count,
            confirmed_count, vhost_count.
        """
        confirmed = [e for e in self._domains.values() if e.source != "auto_expand"]
        vhosts = [e for e in self._domains.values() if e.is_vhost]
        return {
            "primary_domain": self._primary_domain or "",
            "primary_ip": self._primary_ip,
            "domain_count": len(self._domains),
            "confirmed_count": len(confirmed),
            "vhost_count": len(vhosts),
            "all_domains": self.get_all_domains(),
        }

    def reset(self) -> None:
        """Reset all domain state."""
        self._domains.clear()
        self._primary_domain = None
        self._primary_ip = ""
        logger.debug("DomainManager reset")
