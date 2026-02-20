"""core/environment/service_fingerprints.py — Phase 42: Service fingerprint library.

Maps port/service/version tuples to known exploit paths, default
credentials, and recommended enumeration commands. Used by SmartCoach
to prioritize actions based on discovered services.

Complements the existing target_profiler.ServiceFingerprint by providing
a lookup database of common service characteristics.

Author: Phase 42 Contract
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("ariaska.environment.service_fingerprints")


@dataclass
class ServiceProfile:
    """Known characteristics of a service."""
    service_name: str
    common_ports: List[int] = field(default_factory=list)
    default_creds: List[Tuple[str, str]] = field(default_factory=list)
    exploit_paths: List[str] = field(default_factory=list)
    enum_commands: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    cve_patterns: List[str] = field(default_factory=list)
    risk_level: str = "medium"  # low, medium, high, critical


class ServiceFingerprintDB:
    """Phase 42: Service fingerprint lookup database.

    Provides fast lookups by service name, port, or version to retrieve
    known characteristics, default credentials, and recommended
    enumeration strategies.

    Methods:
        lookup_by_service(): Get profile by service name
        lookup_by_port(): Get profiles matching a port number
        get_default_creds(): Get default credentials for a service
        get_exploit_paths(): Get known exploit paths
        get_enum_commands(): Get recommended enumeration commands
        add_profile(): Register a custom service profile
        match_version(): Check version-specific vulnerabilities
    """

    def __init__(self) -> None:
        self._profiles: Dict[str, ServiceProfile] = {}
        self._port_index: Dict[int, List[str]] = {}
        self._load_defaults()
        logger.info(
            "ServiceFingerprintDB initialized with %d profiles",
            len(self._profiles),
        )

    def lookup_by_service(self, service_name: str) -> Optional[ServiceProfile]:
        """Look up profile by service name.

        Args:
            service_name: Service identifier (e.g., "ssh", "http", "smb").

        Returns:
            ServiceProfile if found, None otherwise.
        """
        return self._profiles.get(service_name.lower())

    def lookup_by_port(self, port: int) -> List[ServiceProfile]:
        """Get all service profiles that commonly use a port.

        Args:
            port: Port number to look up.

        Returns:
            List of matching ServiceProfile objects.
        """
        names = self._port_index.get(port, [])
        return [self._profiles[n] for n in names if n in self._profiles]

    def get_default_creds(self, service_name: str) -> List[Tuple[str, str]]:
        """Get default credentials for a service.

        Args:
            service_name: Service identifier.

        Returns:
            List of (username, password) tuples.
        """
        profile = self.lookup_by_service(service_name)
        return profile.default_creds if profile else []

    def get_exploit_paths(self, service_name: str) -> List[str]:
        """Get known exploit paths for a service.

        Args:
            service_name: Service identifier.

        Returns:
            List of exploit path descriptions.
        """
        profile = self.lookup_by_service(service_name)
        return profile.exploit_paths if profile else []

    def get_enum_commands(self, service_name: str) -> List[str]:
        """Get recommended enumeration commands for a service.

        Args:
            service_name: Service identifier.

        Returns:
            List of command template names.
        """
        profile = self.lookup_by_service(service_name)
        return profile.enum_commands if profile else []

    def match_version(
        self, service_name: str, version: str
    ) -> List[str]:
        """Check version-specific CVE patterns.

        Args:
            service_name: Service identifier.
            version: Version string to check.

        Returns:
            List of matching CVE pattern descriptions.
        """
        profile = self.lookup_by_service(service_name)
        if not profile:
            return []
        matches = []
        version_lower = version.lower()
        for pattern in profile.cve_patterns:
            # Simple substring match — could be enhanced with semver
            parts = pattern.split(":", 1)
            if len(parts) == 2:
                ver_match, desc = parts
                if ver_match.lower() in version_lower:
                    matches.append(desc)
        return matches

    def add_profile(self, profile: ServiceProfile) -> None:
        """Register a custom service profile.

        Args:
            profile: The ServiceProfile to register.
        """
        name = profile.service_name.lower()
        self._profiles[name] = profile
        for port in profile.common_ports:
            if port not in self._port_index:
                self._port_index[port] = []
            if name not in self._port_index[port]:
                self._port_index[port].append(name)

    def summary(self) -> Dict[str, Any]:
        """Get database summary.

        Returns:
            Dict with profile count and indexed ports.
        """
        return {
            "total_profiles": len(self._profiles),
            "indexed_ports": len(self._port_index),
            "services": sorted(self._profiles.keys()),
        }

    def _load_defaults(self) -> None:
        """Load built-in service profiles."""
        defaults = [
            ServiceProfile(
                service_name="ssh",
                common_ports=[22],
                default_creds=[("root", "root"), ("admin", "admin"), ("root", "toor")],
                exploit_paths=["CVE-based auth bypass", "user enumeration", "brute force"],
                enum_commands=["ssh_version_scan", "ssh_user_enum", "hydra_ssh"],
                tags={"remote_access", "authentication"},
                cve_patterns=["7.4:OpenSSH 7.4 username enum", "6.6:OpenSSH <6.7 SFTP"],
                risk_level="medium",
            ),
            ServiceProfile(
                service_name="http",
                common_ports=[80, 8080, 8000, 8443],
                default_creds=[("admin", "admin"), ("admin", "password")],
                exploit_paths=["directory traversal", "RCE via webapp", "file upload"],
                enum_commands=["gobuster_dir", "nikto_scan", "curl_headers", "wpscan"],
                tags={"web", "application"},
                cve_patterns=[],
                risk_level="high",
            ),
            ServiceProfile(
                service_name="https",
                common_ports=[443, 8443],
                default_creds=[],
                exploit_paths=["SSL/TLS vuln", "web app exploit", "certificate issues"],
                enum_commands=["sslscan", "nikto_scan", "gobuster_dir"],
                tags={"web", "encrypted"},
                risk_level="high",
            ),
            ServiceProfile(
                service_name="ftp",
                common_ports=[21],
                default_creds=[("anonymous", ""), ("ftp", "ftp"), ("admin", "admin")],
                exploit_paths=["anonymous login", "version exploit", "directory traversal"],
                enum_commands=["ftp_anon_check", "ftp_version_scan", "hydra_ftp"],
                tags={"file_transfer", "authentication"},
                cve_patterns=["2.3.4:vsftpd 2.3.4 backdoor"],
                risk_level="high",
            ),
            ServiceProfile(
                service_name="smb",
                common_ports=[139, 445],
                default_creds=[("guest", ""), ("admin", "admin")],
                exploit_paths=["EternalBlue", "null session", "share enum"],
                enum_commands=["enum4linux", "smbclient_list", "smbmap", "crackmapexec"],
                tags={"file_sharing", "windows"},
                cve_patterns=["1.0:MS17-010 EternalBlue", "3.1.1:SMBGhost"],
                risk_level="critical",
            ),
            ServiceProfile(
                service_name="mysql",
                common_ports=[3306],
                default_creds=[("root", ""), ("root", "root"), ("root", "mysql")],
                exploit_paths=["auth bypass", "UDF exploit", "file read"],
                enum_commands=["mysql_version", "mysql_login", "mysql_enum"],
                tags={"database", "authentication"},
                cve_patterns=[],
                risk_level="high",
            ),
            ServiceProfile(
                service_name="postgresql",
                common_ports=[5432],
                default_creds=[("postgres", "postgres"), ("postgres", "")],
                exploit_paths=["auth bypass", "command exec", "file read/write"],
                enum_commands=["psql_version", "psql_login"],
                tags={"database", "authentication"},
                risk_level="high",
            ),
            ServiceProfile(
                service_name="redis",
                common_ports=[6379],
                default_creds=[],
                exploit_paths=["unauthenticated access", "RCE via module", "SSH key write"],
                enum_commands=["redis_info", "redis_cli"],
                tags={"database", "nosql"},
                risk_level="critical",
            ),
            ServiceProfile(
                service_name="telnet",
                common_ports=[23],
                default_creds=[("admin", "admin"), ("root", "root")],
                exploit_paths=["cleartext sniffing", "brute force"],
                enum_commands=["telnet_banner", "hydra_telnet"],
                tags={"remote_access", "legacy"},
                risk_level="high",
            ),
            ServiceProfile(
                service_name="snmp",
                common_ports=[161, 162],
                default_creds=[],
                exploit_paths=["community string brute", "info disclosure"],
                enum_commands=["snmpwalk", "snmp_enum", "onesixtyone"],
                tags={"network", "management"},
                cve_patterns=[],
                risk_level="medium",
            ),
            ServiceProfile(
                service_name="rdp",
                common_ports=[3389],
                default_creds=[("administrator", "password")],
                exploit_paths=["BlueKeep CVE-2019-0708", "brute force"],
                enum_commands=["rdp_check", "hydra_rdp"],
                tags={"remote_access", "windows"},
                cve_patterns=["pre-NLA:CVE-2019-0708 BlueKeep"],
                risk_level="critical",
            ),
            ServiceProfile(
                service_name="vnc",
                common_ports=[5900, 5901],
                default_creds=[],
                exploit_paths=["auth bypass", "brute force"],
                enum_commands=["vnc_check", "hydra_vnc"],
                tags={"remote_access", "gui"},
                risk_level="high",
            ),
            ServiceProfile(
                service_name="smtp",
                common_ports=[25, 587],
                default_creds=[],
                exploit_paths=["user enum VRFY/EXPN", "open relay"],
                enum_commands=["smtp_user_enum", "smtp_vrfy"],
                tags={"email", "network"},
                risk_level="medium",
            ),
            ServiceProfile(
                service_name="dns",
                common_ports=[53],
                default_creds=[],
                exploit_paths=["zone transfer", "cache poisoning"],
                enum_commands=["dig_axfr", "dnsenum", "dnsrecon"],
                tags={"network", "infrastructure"},
                risk_level="medium",
            ),
            ServiceProfile(
                service_name="ldap",
                common_ports=[389, 636],
                default_creds=[],
                exploit_paths=["anonymous bind", "injection"],
                enum_commands=["ldapsearch", "ldap_enum"],
                tags={"directory", "authentication"},
                risk_level="high",
            ),
        ]
        for profile in defaults:
            self.add_profile(profile)
