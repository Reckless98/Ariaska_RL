"""
Target Profiler — Dynamic Target Intelligence for Ariaska_RL.

Phase 9.2: Generalizes beyond MS2/MS3 to support arbitrary targets
(HackTheBox, CTF, cloud, AD environments) by profiling scan results
and dynamically selecting relevant knowledge packs.

Phase 9.3: Unknown-target generalization with chain-of-thought reasoning.
  - UnknownTargetStrategy: Analyzes any target via service archetypes and
    chain-of-thought logic to produce attack recommendations even when the
    target type cannot be classified as a known environment.
  - GeneralizationPlaybook: Builds generic multi-phase attack plans from
    observed service archetypes (auth, web, database, remote_exec, etc.)

Architecture:
  - TargetProfile: Structured representation of a target's attack surface
  - ServiceArchetype: Categorizes services into attack-relevant archetypes
  - TargetProfiler: Builds profiles from nmap/scan output and KG data
  - UnknownTargetStrategy: Chain-of-thought reasoning for unclassified targets
  - GeneralizationPlaybook: Fallback attack plan builder from service archetypes

Usage:
    from core.knowledge.target_profiler import TargetProfiler
    profiler = TargetProfiler(knowledge_graph=kg, retriever=kr)
    profile = profiler.profile_from_nmap(nmap_output)
    knowledge = profiler.select_knowledge(profile)

    # Unknown-target reasoning:
    strategy = UnknownTargetStrategy()
    assessment = strategy.assess(profile)
    playbook = GeneralizationPlaybook.from_profile(profile)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from core.knowledge.kg_manager import KnowledgeGraph
    from data.knowledge_retriever import KnowledgeRetriever

logger = logging.getLogger("ariaska.target_profiler")


# ─── Target Classification ──────────────────────────────────────────────────

class TargetType:
    """Known target environment types."""
    METASPLOITABLE2 = "metasploitable2"
    METASPLOITABLE3 = "metasploitable3"
    HACKTHEBOX = "hackthebox"
    CTF = "ctf"
    ACTIVE_DIRECTORY = "active_directory"
    CLOUD = "cloud"
    IOT = "iot"
    WEBAPP = "webapp"
    GENERIC_LINUX = "generic_linux"
    GENERIC_WINDOWS = "generic_windows"
    UNKNOWN = "unknown"


# ─── Service Archetypes ─────────────────────────────────────────────────────

class ServiceArchetype(Enum):
    """
    High-level service categories for generalized attack reasoning.
    
    Every open service maps to one or more archetypes, enabling
    chain-of-thought attack planning without target recognition.
    """
    AUTH = "auth"               # Login-capable: SSH, Telnet, FTP, RDP
    WEB = "web"                 # HTTP/HTTPS services (Apache, Nginx, IIS, etc.)
    DATABASE = "database"       # MySQL, PostgreSQL, MSSQL, MongoDB, Redis
    FILE_SHARING = "file_share" # SMB, NFS, FTP (data), rsync
    REMOTE_EXEC = "remote_exec" # RPC, rexec, rlogin, WinRM, Java RMI
    MAIL = "mail"               # SMTP, POP3, IMAP
    DNS = "dns"                 # DNS
    DIRECTORY = "directory"     # LDAP, AD, Kerberos
    MONITORING = "monitoring"   # SNMP, Zabbix, Nagios
    MESSAGING = "messaging"     # IRC, XMPP, AMQP
    VNC_RDP = "vnc_rdp"         # VNC, RDP — GUI remote access
    CONTAINER = "container"     # Docker API, Kubernetes API
    CI_CD = "ci_cd"             # Jenkins, GitLab, Bamboo
    CUSTOM_APP = "custom_app"   # Non-standard ports, likely custom applications


# Comprehensive port → archetype mapping
_PORT_ARCHETYPE_MAP: Dict[int, List[ServiceArchetype]] = {
    # Auth services
    21: [ServiceArchetype.AUTH, ServiceArchetype.FILE_SHARING],
    22: [ServiceArchetype.AUTH],
    23: [ServiceArchetype.AUTH, ServiceArchetype.REMOTE_EXEC],
    513: [ServiceArchetype.AUTH, ServiceArchetype.REMOTE_EXEC],
    3389: [ServiceArchetype.VNC_RDP, ServiceArchetype.AUTH],
    # Web
    80: [ServiceArchetype.WEB],
    443: [ServiceArchetype.WEB],
    8080: [ServiceArchetype.WEB],
    8443: [ServiceArchetype.WEB],
    8000: [ServiceArchetype.WEB],
    8888: [ServiceArchetype.WEB],
    3000: [ServiceArchetype.WEB],
    5000: [ServiceArchetype.WEB],
    8180: [ServiceArchetype.WEB],
    9090: [ServiceArchetype.WEB],
    # Database
    3306: [ServiceArchetype.DATABASE],
    5432: [ServiceArchetype.DATABASE],
    1433: [ServiceArchetype.DATABASE],
    1521: [ServiceArchetype.DATABASE],
    27017: [ServiceArchetype.DATABASE],
    6379: [ServiceArchetype.DATABASE],
    9200: [ServiceArchetype.DATABASE],          # Elasticsearch
    5984: [ServiceArchetype.DATABASE],          # CouchDB
    # File sharing
    139: [ServiceArchetype.FILE_SHARING],
    445: [ServiceArchetype.FILE_SHARING],
    2049: [ServiceArchetype.FILE_SHARING],
    873: [ServiceArchetype.FILE_SHARING],       # rsync
    69: [ServiceArchetype.FILE_SHARING],        # TFTP
    # Remote exec
    512: [ServiceArchetype.REMOTE_EXEC],
    514: [ServiceArchetype.REMOTE_EXEC],
    1099: [ServiceArchetype.REMOTE_EXEC],       # Java RMI
    1524: [ServiceArchetype.REMOTE_EXEC],       # ingreslock backdoor
    5985: [ServiceArchetype.REMOTE_EXEC],       # WinRM HTTP
    5986: [ServiceArchetype.REMOTE_EXEC],       # WinRM HTTPS
    # Mail
    25: [ServiceArchetype.MAIL],
    110: [ServiceArchetype.MAIL],
    143: [ServiceArchetype.MAIL],
    465: [ServiceArchetype.MAIL],
    587: [ServiceArchetype.MAIL],
    993: [ServiceArchetype.MAIL],
    995: [ServiceArchetype.MAIL],
    # DNS
    53: [ServiceArchetype.DNS],
    # Directory / AD
    88: [ServiceArchetype.DIRECTORY],
    389: [ServiceArchetype.DIRECTORY],
    636: [ServiceArchetype.DIRECTORY],
    3268: [ServiceArchetype.DIRECTORY],
    # Monitoring
    161: [ServiceArchetype.MONITORING],
    162: [ServiceArchetype.MONITORING],
    # Messaging
    6667: [ServiceArchetype.MESSAGING],
    6697: [ServiceArchetype.MESSAGING],
    5222: [ServiceArchetype.MESSAGING],
    5672: [ServiceArchetype.MESSAGING],         # AMQP / RabbitMQ
    # VNC
    5900: [ServiceArchetype.VNC_RDP],
    5901: [ServiceArchetype.VNC_RDP],
    # Container / CI-CD
    2375: [ServiceArchetype.CONTAINER],
    2376: [ServiceArchetype.CONTAINER],
    6443: [ServiceArchetype.CONTAINER],         # Kubernetes API
    8081: [ServiceArchetype.CI_CD],             # Nexus / CI
    50000: [ServiceArchetype.CI_CD],            # Jenkins agent
}

# Service name → archetype for fuzzy matching
_SERVICE_NAME_ARCHETYPE: Dict[str, ServiceArchetype] = {
    "ssh": ServiceArchetype.AUTH,
    "ftp": ServiceArchetype.AUTH,
    "telnet": ServiceArchetype.AUTH,
    "http": ServiceArchetype.WEB,
    "https": ServiceArchetype.WEB,
    "http-proxy": ServiceArchetype.WEB,
    "mysql": ServiceArchetype.DATABASE,
    "postgresql": ServiceArchetype.DATABASE,
    "ms-sql-s": ServiceArchetype.DATABASE,
    "oracle-tns": ServiceArchetype.DATABASE,
    "mongodb": ServiceArchetype.DATABASE,
    "redis": ServiceArchetype.DATABASE,
    "microsoft-ds": ServiceArchetype.FILE_SHARING,
    "netbios-ssn": ServiceArchetype.FILE_SHARING,
    "nfs": ServiceArchetype.FILE_SHARING,
    "smb": ServiceArchetype.FILE_SHARING,
    "smtp": ServiceArchetype.MAIL,
    "pop3": ServiceArchetype.MAIL,
    "imap": ServiceArchetype.MAIL,
    "domain": ServiceArchetype.DNS,
    "kerberos-sec": ServiceArchetype.DIRECTORY,
    "ldap": ServiceArchetype.DIRECTORY,
    "snmp": ServiceArchetype.MONITORING,
    "irc": ServiceArchetype.MESSAGING,
    "vnc": ServiceArchetype.VNC_RDP,
    "ms-wbt-server": ServiceArchetype.VNC_RDP,
    "exec": ServiceArchetype.REMOTE_EXEC,
    "login": ServiceArchetype.REMOTE_EXEC,
    "shell": ServiceArchetype.REMOTE_EXEC,
    "rmiregistry": ServiceArchetype.REMOTE_EXEC,
    "docker": ServiceArchetype.CONTAINER,
    "kubernetes": ServiceArchetype.CONTAINER,
}


def classify_service(svc: ServiceFingerprint) -> List[ServiceArchetype]:
    """Classify a service into one or more archetypes."""
    archetypes = []

    # Check port map first
    if svc.port in _PORT_ARCHETYPE_MAP:
        archetypes.extend(_PORT_ARCHETYPE_MAP[svc.port])

    # Check service name
    svc_lower = svc.service.lower() if svc.service else ""
    if svc_lower in _SERVICE_NAME_ARCHETYPE:
        arch = _SERVICE_NAME_ARCHETYPE[svc_lower]
        if arch not in archetypes:
            archetypes.append(arch)

    # Product-based heuristics
    product_lower = svc.product.lower() if svc.product else ""
    if product_lower:
        for name, arch in _SERVICE_NAME_ARCHETYPE.items():
            if name in product_lower and arch not in archetypes:
                archetypes.append(arch)

    # Unknown port with no match → custom app
    if not archetypes:
        archetypes.append(ServiceArchetype.CUSTOM_APP)

    return archetypes


@dataclass
class ServiceFingerprint:
    """A detected service on the target."""
    port: int
    protocol: str = "tcp"
    service: str = ""
    version: str = ""
    product: str = ""
    os_guess: str = ""
    cpe: str = ""
    state: str = "open"
    banner: str = ""
    
    @property
    def is_web(self) -> bool:
        return self.port in (80, 443, 8080, 8443, 8180, 3000, 5000) or \
               self.service in ("http", "https", "http-proxy")
    
    @property
    def is_auth(self) -> bool:
        return self.service in ("ssh", "telnet", "ftp", "rdp", "vnc", "smb") or \
               self.port in (22, 23, 21, 3389, 5900, 445)


@dataclass
class TargetProfile:
    """Complete intelligence profile for a target."""
    target_ip: str = ""
    hostname: str = ""
    os_family: str = "unknown"       # linux, windows, unknown
    os_version: str = ""
    target_type: str = TargetType.UNKNOWN
    confidence: float = 0.0          # 0-1 confidence in classification
    
    # Discovered services
    services: List[ServiceFingerprint] = field(default_factory=list)
    open_ports: Set[int] = field(default_factory=set)
    
    # Attack surface analysis
    web_surfaces: List[Dict[str, Any]] = field(default_factory=list)
    auth_services: List[Dict[str, Any]] = field(default_factory=list)
    database_services: List[Dict[str, Any]] = field(default_factory=list)
    
    # Knowledge-matched vulnerabilities
    matched_cves: List[str] = field(default_factory=list)
    matched_exploits: List[str] = field(default_factory=list)
    recommended_techniques: List[str] = field(default_factory=list)
    
    # Suggested attack chains (ordered)
    attack_chains: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    scan_source: str = "manual"  # "nmap", "manual", "cached"
    
    # Phase 9.3: Archetype analysis for unknown-target generalization
    service_archetypes: Dict[str, List[int]] = field(default_factory=dict)
    # Maps ServiceArchetype.value → list of ports matching that archetype
    
    generalization_assessment: Optional[Dict[str, Any]] = field(default=None)
    # Chain-of-thought assessment from UnknownTargetStrategy
    
    @property
    def has_web(self) -> bool:
        return any(s.is_web for s in self.services)
    
    @property
    def has_ad(self) -> bool:
        """Check for Active Directory indicators."""
        ad_ports = {88, 135, 139, 389, 445, 464, 636, 3268, 3269}
        return len(ad_ports & self.open_ports) >= 3
    
    @property
    def service_names(self) -> List[str]:
        return [s.service for s in self.services if s.service]
    
    @property
    def archetype_summary(self) -> Dict[str, int]:
        """Count of services per archetype — used for reasoning."""
        counts: Dict[str, int] = {}
        for arch_name, ports in self.service_archetypes.items():
            counts[arch_name] = len(ports)
        return counts
    
    @property
    def dominant_archetypes(self) -> List[str]:
        """Top archetypes by service count, descending."""
        summary = self.archetype_summary
        return sorted(summary, key=summary.get, reverse=True)  # type: ignore[arg-type]
    
    def to_prompt_fragment(self, max_chars: int = 2000) -> str:
        """Convert to a prompt-injectable string for LLM context."""
        lines = [
            f"=== TARGET PROFILE: {self.target_ip} ===",
            f"OS: {self.os_family} {self.os_version}".strip(),
            f"Type: {self.target_type} (conf={self.confidence:.0%})",
            f"Open Ports: {sorted(self.open_ports)[:20]}",
        ]
        
        if self.services:
            lines.append("\nServices:")
            for svc in self.services[:15]:
                lines.append(f"  {svc.port}/{svc.protocol} {svc.service} {svc.version}".strip())
        
        if self.matched_cves:
            lines.append(f"\nMatched CVEs: {', '.join(self.matched_cves[:10])}")
        
        if self.matched_exploits:
            lines.append(f"\nExploits: {', '.join(self.matched_exploits[:5])}")
        
        if self.attack_chains:
            lines.append("\nRecommended Attack Chains:")
            for i, chain in enumerate(self.attack_chains[:3], 1):
                lines.append(f"  {i}. {chain.get('name', '?')}: {chain.get('summary', '')[:100]}")
        
        result = "\n".join(lines)
        return result[:max_chars]


# ─── Target Profiler ────────────────────────────────────────────────────────

class TargetProfiler:
    """
    Builds target profiles from scan output and enriches them with KG data.
    
    Supports automatic target classification (MS2, MS3, HTB, AD, cloud, etc.)
    and dynamic knowledge selection based on the target's attack surface.
    """

    # MS2 fingerprint: these ports together strongly indicate Metasploitable 2
    MS2_SIGNATURE_PORTS = {21, 22, 23, 25, 80, 139, 445, 512, 513, 514,
                           1099, 1524, 2049, 3306, 5432, 5900, 6667, 8180}
    
    # MS3 fingerprint
    MS3_SIGNATURE_PORTS = {22, 80, 3000, 3306, 3500, 8080, 8181, 8484, 9200}
    
    # AD indicators
    AD_PORTS = {88, 135, 139, 389, 445, 464, 636, 3268, 3269}

    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        retriever: Optional[KnowledgeRetriever] = None,
    ):
        self.kg = knowledge_graph
        self.kr = retriever

    def profile_from_nmap(self, nmap_output: str, target_ip: str = "") -> TargetProfile:
        """
        Build a target profile from nmap scan output.
        
        Parses nmap text output to extract services, then classifies
        the target and enriches with knowledge base data.
        """
        profile = TargetProfile(target_ip=target_ip, scan_source="nmap")
        
        # Parse nmap output
        services = self._parse_nmap_output(nmap_output)
        profile.services = services
        profile.open_ports = {s.port for s in services}
        
        # Detect OS
        profile.os_family = self._detect_os(nmap_output, services)
        
        # Classify target type
        profile.target_type, profile.confidence = self._classify_target(profile)
        
        # Enrich with knowledge base data
        self._enrich_profile(profile)
        
        return profile

    def profile_from_state(self, state: Dict[str, Any]) -> TargetProfile:
        """
        Build a target profile from CyberEnvironment state dict.
        
        Used during training to create profiles from environment state
        without needing raw nmap output.
        """
        profile = TargetProfile(
            target_ip=state.get("target_ip", ""),
            scan_source="state",
        )
        
        # Extract ports from state flags
        for key, val in state.items():
            if key.startswith("port_") and val:
                try:
                    port = int(key.replace("port_", ""))
                    profile.open_ports.add(port)
                    profile.services.append(ServiceFingerprint(port=port))
                except ValueError:
                    pass
        
        # Extract discovered services
        discovered_services = state.get("discovered_services", [])
        if isinstance(discovered_services, (list, set)):
            for svc_str in discovered_services:
                if isinstance(svc_str, str):
                    # Parse "service:port" or just "service"
                    parts = svc_str.split(":")
                    svc_name = parts[0]
                    svc_port = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                    if svc_port:
                        profile.services.append(ServiceFingerprint(
                            port=svc_port, service=svc_name,
                        ))
        
        profile.os_family = state.get("platform", state.get("os", "unknown"))
        profile.target_type, profile.confidence = self._classify_target(profile)
        self._enrich_profile(profile)
        
        return profile

    # ─── Parsing ─────────────────────────────────────────────────────────

    def _parse_nmap_output(self, output: str) -> List[ServiceFingerprint]:
        """Parse nmap -sV output into ServiceFingerprint objects."""
        services = []
        
        # Match lines like: "21/tcp  open  ftp  vsftpd 2.3.4"
        port_re = re.compile(
            r'(\d+)/(tcp|udp)\s+'
            r'(open|filtered|closed)\s+'
            r'(\S+)\s*'
            r'(.*?)$',
            re.MULTILINE,
        )
        
        for m in port_re.finditer(output):
            port = int(m.group(1))
            protocol = m.group(2)
            state = m.group(3)
            service = m.group(4)
            version_info = m.group(5).strip() if m.group(5) else ""
            
            if state != "open":
                continue
            
            svc = ServiceFingerprint(
                port=port,
                protocol=protocol,
                service=service,
                version=version_info,
                state=state,
            )
            
            # Extract product/version from version string
            if version_info:
                # e.g., "vsftpd 2.3.4" or "Apache httpd 2.2.8"
                parts = version_info.split()
                if parts:
                    svc.product = parts[0]
                    if len(parts) > 1:
                        svc.version = " ".join(parts[1:])
            
            services.append(svc)
        
        return services

    def _detect_os(self, nmap_output: str, services: List[ServiceFingerprint]) -> str:
        """Detect OS family from nmap output and service fingerprints."""
        output_lower = nmap_output.lower()
        
        # Direct OS detection from nmap
        if "linux" in output_lower or "ubuntu" in output_lower or "debian" in output_lower:
            return "linux"
        if "windows" in output_lower or "microsoft" in output_lower:
            return "windows"
        
        # Infer from services
        linux_indicators = {"openssh", "vsftpd", "proftpd", "apache", "nginx"}
        windows_indicators = {"iis", "rdp", "ms-wbt-server", "microsoft-ds"}
        
        linux_score = sum(1 for s in services if s.product.lower() in linux_indicators)
        win_score = sum(1 for s in services if s.product.lower() in windows_indicators)
        
        if linux_score > win_score:
            return "linux"
        elif win_score > linux_score:
            return "windows"
        return "unknown"

    # ─── Classification ──────────────────────────────────────────────────

    def _classify_target(self, profile: TargetProfile) -> Tuple[str, float]:
        """
        Classify target type based on service fingerprint.
        
        Returns (target_type, confidence) tuple.
        """
        ports = profile.open_ports
        
        # MS2 detection: many of the signature ports overlap
        ms2_overlap = len(ports & self.MS2_SIGNATURE_PORTS)
        if ms2_overlap >= 10:
            return TargetType.METASPLOITABLE2, min(ms2_overlap / 15, 0.95)
        
        # MS3 detection
        ms3_overlap = len(ports & self.MS3_SIGNATURE_PORTS)
        if ms3_overlap >= 5:
            return TargetType.METASPLOITABLE3, min(ms3_overlap / 8, 0.9)
        
        # AD detection
        ad_overlap = len(ports & self.AD_PORTS)
        if ad_overlap >= 4:
            return TargetType.ACTIVE_DIRECTORY, min(ad_overlap / 7, 0.85)
        
        # Web-heavy target (HTB style)
        web_ports = {p for p in ports if p in (80, 443, 8080, 8443, 3000, 5000, 8000)}
        if len(web_ports) >= 1 and len(ports) <= 5:
            return TargetType.HACKTHEBOX, 0.4  # Low confidence — could be any web target
        
        # Classify by OS
        if profile.os_family == "linux":
            return TargetType.GENERIC_LINUX, 0.3
        elif profile.os_family == "windows":
            return TargetType.GENERIC_WINDOWS, 0.3
        
        return TargetType.UNKNOWN, 0.1

    # ─── Knowledge Enrichment ────────────────────────────────────────────

    def _enrich_profile(self, profile: TargetProfile):
        """Enrich profile with knowledge base and KG data."""
        # Phase 9.3: Classify services into archetypes
        self._classify_archetypes(profile)
        
        if self.kr:
            self._enrich_from_retriever(profile)
        if self.kg:
            self._enrich_from_kg(profile)
        self._build_attack_chains(profile)
        
        # Phase 9.3: If target is UNKNOWN or low-confidence, run generalization
        if profile.target_type == TargetType.UNKNOWN or profile.confidence < 0.3:
            strategy = UnknownTargetStrategy()
            profile.generalization_assessment = strategy.assess(profile)
            # Inject generalized chains
            gen_playbook = GeneralizationPlaybook.from_profile(profile)
            for chain in gen_playbook.chains:
                if chain not in profile.attack_chains:
                    profile.attack_chains.append(chain)

    def _classify_archetypes(self, profile: TargetProfile):
        """Classify each service into archetypes and store on profile."""
        archetype_ports: Dict[str, List[int]] = {}
        for svc in profile.services:
            archetypes = classify_service(svc)
            for arch in archetypes:
                key = arch.value
                if key not in archetype_ports:
                    archetype_ports[key] = []
                if svc.port not in archetype_ports[key]:
                    archetype_ports[key].append(svc.port)
        profile.service_archetypes = archetype_ports

    def _enrich_from_retriever(self, profile: TargetProfile):
        """Enrich from KnowledgeRetriever."""
        try:
            kr = self.kr
            for svc in profile.services:
                # Get CVEs for this service
                if svc.product:
                    search_results = kr.search(
                        f"{svc.product} {svc.version}".strip(), max_results=5,
                    )
                    for score, entry, cat in search_results:
                        if cat == "cve":
                            cve_id = entry.get("cve_id", "")
                            if cve_id and cve_id not in profile.matched_cves:
                                profile.matched_cves.append(cve_id)
                        elif cat == "exploit_module":
                            mod_name = entry.get("name", entry.get("module_name", ""))
                            if mod_name and mod_name not in profile.matched_exploits:
                                profile.matched_exploits.append(mod_name)
                
                # Get kill chains
                chains = kr.get_kill_chain(svc.port, max_results=2)
                for chain in chains:
                    profile.attack_chains.append({
                        "name": chain.get("chain_name", f"chain_{svc.port}"),
                        "target_port": svc.port,
                        "steps": chain.get("steps", []),
                        "summary": chain.get("reasoning", "")[:200],
                        "source": "knowledge_base",
                    })
        except Exception as e:
            logger.debug(f"KR enrichment failed: {e}")

    def _enrich_from_kg(self, profile: TargetProfile):
        """Enrich from KnowledgeGraph."""
        try:
            kg = self.kg
            for svc in profile.services:
                # Find exploit nodes connected to this service
                exploits = kg.get_exploits_for_service(svc.service, svc.port)
                for exploit in exploits[:5]:
                    eid = exploit.get("node_id", exploit.get("id", ""))
                    if eid and eid not in profile.matched_exploits:
                        profile.matched_exploits.append(eid)
                
                # Find relevant techniques
                techniques = kg.get_techniques_for_phase(
                    "EXPLOITATION" if svc.port not in (22, 23) else "ENUMERATION",
                )
                for tech in techniques[:3]:
                    tid = tech.get("node_id", tech.get("id", ""))
                    if tid and tid not in profile.recommended_techniques:
                        profile.recommended_techniques.append(tid)
        except Exception as e:
            logger.debug(f"KG enrichment failed: {e}")

    def _build_attack_chains(self, profile: TargetProfile):
        """Build recommended attack chains based on target profile."""
        # Only add auto-generated chains if we don't already have KB chains
        if len(profile.attack_chains) >= 3:
            return
        
        # Generate chains based on target type
        if profile.target_type == TargetType.METASPLOITABLE2:
            profile.attack_chains.insert(0, {
                "name": "MS2 Instant Root",
                "summary": "telnet to ingreslock (1524) → instant root shell",
                "steps": [
                    {"step": 1, "command": f"nmap -sV {profile.target_ip}"},
                    {"step": 2, "command": f"telnet {profile.target_ip} 1524"},
                ],
                "source": "profiler",
            })
        
        elif profile.target_type == TargetType.ACTIVE_DIRECTORY:
            profile.attack_chains.insert(0, {
                "name": "AD Standard Chain",
                "summary": "Enumerate → AS-REP Roast → Kerberoast → Pass-the-Hash",
                "steps": [
                    {"step": 1, "command": "enum4linux -a {target}"},
                    {"step": 2, "command": "GetNPUsers.py domain/ -no-pass -usersfile users.txt"},
                    {"step": 3, "command": "GetUserSPNs.py domain/user:pass -request"},
                    {"step": 4, "command": "secretsdump.py domain/user:pass@{target}"},
                ],
                "source": "profiler",
            })

    # ─── Knowledge Selection ─────────────────────────────────────────────

    def select_knowledge(self, profile: TargetProfile) -> Dict[str, Any]:
        """
        Select relevant knowledge subsets based on target profile.
        
        Returns a dict with selected knowledge categories and their content,
        optimized for the specific target type.
        
        Phase 9.3: For unknown targets, includes archetype-based reasoning
        and generalization assessment to guide the attack.
        """
        knowledge: Dict[str, Any] = {
            "target_type": profile.target_type,
            "os_family": profile.os_family,
            "relevant_techniques": [],
            "relevant_exploits": [],
            "relevant_cves": profile.matched_cves[:20],
            "attack_chains": profile.attack_chains[:5],
            "privesc_techniques": [],
            "credential_targets": [],
        }
        
        # Phase 9.3: Include archetype analysis for all targets
        if profile.service_archetypes:
            knowledge["service_archetypes"] = profile.archetype_summary
            knowledge["dominant_archetypes"] = profile.dominant_archetypes[:5]
        
        # Phase 9.3: Include generalization assessment for unknown targets
        if profile.generalization_assessment:
            knowledge["generalization"] = profile.generalization_assessment
        
        # Select privesc techniques based on OS
        if self.kr:
            try:
                if profile.os_family == "linux":
                    knowledge["privesc_techniques"] = self.kr.get_privesc("linux", max_results=10)
                    knowledge["gtfobins"] = self.kr.get_gtfobins_for_privesc(max_results=10)
                elif profile.os_family == "windows":
                    knowledge["privesc_techniques"] = self.kr.get_privesc("windows", max_results=10)
                
                # Get credential targets for auth services
                for svc in profile.services:
                    if svc.is_auth:
                        creds = self.kr.get_credentials(svc.service)
                        if creds:
                            knowledge["credential_targets"].extend(creds[:3])
                
                # AD-specific knowledge
                if profile.has_ad or profile.target_type == TargetType.ACTIVE_DIRECTORY:
                    knowledge["ad_attacks"] = self.kr.get_ad_attacks(max_results=15)
                
                # Cloud knowledge
                if profile.target_type == TargetType.CLOUD:
                    knowledge["cloud_attacks"] = self.kr.get_cloud_attacks(max_results=15)
                
                # Exploit modules for matched CVEs
                for cve in profile.matched_cves[:10]:
                    mods = self.kr.get_exploit_modules(cve=cve, max_results=3)
                    knowledge["relevant_exploits"].extend(mods)
            except Exception as e:
                logger.debug(f"Knowledge selection from KR failed: {e}")
        
        return knowledge

    def __repr__(self) -> str:
        return f"<TargetProfiler kg={self.kg is not None} kr={self.kr is not None}>"


# ═══════════════════════════════════════════════════════════════════════════
# Phase 9.3: Unknown-Target Generalization
# ═══════════════════════════════════════════════════════════════════════════


# ─── Archetype Attack Reasoning ──────────────────────────────────────────

# For each service archetype, define:
#   - attack_priority: How urgent to attack this archetype (1-5)
#   - default_tools: Commands to try first
#   - credential_targets: What cred types to look for
#   - privesc_potential: Can this archetype lead to privesc?
#   - reasoning: Human-readable chain-of-thought for this archetype

_ARCHETYPE_REASONING: Dict[str, Dict[str, Any]] = {
    ServiceArchetype.AUTH.value: {
        "attack_priority": 4,
        "reasoning": (
            "AUTH services accept credentials. STEP 1: Try default/common credentials "
            "(admin:admin, root:root, service-specific defaults). STEP 2: If banner reveals "
            "software version, search for auth bypass CVEs. STEP 3: Brute-force with targeted "
            "wordlist if no quick win. STEP 4: If SSH, try key-based auth or known weak keys."
        ),
        "default_tools": [
            "hydra -l admin -P /usr/share/wordlists/rockyou.txt {proto}://{target}",
            "nmap --script {proto}-brute -p {port} {target}",
            "medusa -h {target} -u root -P /usr/share/wordlists/rockyou.txt -M {proto}",
        ],
        "credential_targets": ["password", "ssh_key", "token"],
        "privesc_potential": True,
        "phase": "EXPLOITATION",
    },
    ServiceArchetype.WEB.value: {
        "attack_priority": 5,
        "reasoning": (
            "WEB services are the richest attack surface. STEP 1: Enumerate directories and files "
            "(gobuster/dirb/feroxbuster). STEP 2: Identify technologies (whatweb/wappalyzer). "
            "STEP 3: Check for known web app vulnerabilities (DVWA, WordPress, phpMyAdmin). "
            "STEP 4: Test for injection points (SQLi, XSS, SSTI, LFI/RFI). STEP 5: Look for "
            "file upload points → web shell. STEP 6: Check for API endpoints and auth bypass."
        ),
        "default_tools": [
            "gobuster dir -u http://{target}:{port} -w /usr/share/wordlists/dirb/common.txt",
            "nikto -h http://{target}:{port}",
            "whatweb http://{target}:{port}",
            "sqlmap -u 'http://{target}:{port}/page?id=1' --batch",
            "wfuzz -c -z file,/usr/share/wordlists/dirb/common.txt http://{target}:{port}/FUZZ",
        ],
        "credential_targets": ["password", "session_cookie", "api_key", "jwt_token"],
        "privesc_potential": True,
        "phase": "ENUMERATION",
    },
    ServiceArchetype.DATABASE.value: {
        "attack_priority": 5,
        "reasoning": (
            "DATABASES often have weak or default credentials and can escalate to OS command "
            "execution. STEP 1: Try default credentials (root with no password for MySQL, "
            "postgres:postgres for PostgreSQL, sa:sa for MSSQL). STEP 2: If authenticated, "
            "enumerate databases, tables, and users. STEP 3: Attempt command execution "
            "(MySQL INTO OUTFILE, PostgreSQL COPY FROM PROGRAM, MSSQL xp_cmdshell). "
            "STEP 4: Extract credential hashes and sensitive data."
        ),
        "default_tools": [
            "mysql -h {target} -u root --password=''",
            "psql -h {target} -U postgres -W",
            "nmap --script mysql-enum,mysql-brute -p {port} {target}",
            "nmap --script pgsql-brute -p {port} {target}",
        ],
        "credential_targets": ["password", "db_hash", "connection_string"],
        "privesc_potential": True,
        "phase": "EXPLOITATION",
    },
    ServiceArchetype.FILE_SHARING.value: {
        "attack_priority": 4,
        "reasoning": (
            "FILE SHARING services expose internal data and often have weak ACLs. "
            "STEP 1: Enumerate shares/exports (smbclient -L, showmount -e). "
            "STEP 2: Check for anonymous/guest access. STEP 3: Search writable shares "
            "for config files, credentials, SSH keys. STEP 4: For NFS, check if root_squash "
            "is disabled → mount and plant SSH keys for root access. STEP 5: For SMB, check "
            "for known exploits (EternalBlue, Samba usermap_script)."
        ),
        "default_tools": [
            "smbclient -L //{target} -N",
            "enum4linux -a {target}",
            "showmount -e {target}",
            "smbmap -H {target}",
            "nmap --script smb-enum-shares -p {port} {target}",
        ],
        "credential_targets": ["password", "ntlm_hash", "ssh_key"],
        "privesc_potential": True,
        "phase": "ENUMERATION",
    },
    ServiceArchetype.REMOTE_EXEC.value: {
        "attack_priority": 5,
        "reasoning": (
            "REMOTE EXEC services provide direct command execution — highest value targets. "
            "STEP 1: Check if service requires authentication (rexec/rlogin often don't). "
            "STEP 2: For Java RMI, attempt deserialization attacks. STEP 3: For known "
            "backdoors (ingreslock 1524), connect directly. STEP 4: For WinRM, try "
            "credential spraying. These services often give instant shell access."
        ),
        "default_tools": [
            "rlogin -l root {target}",
            "telnet {target} {port}",
            "nmap --script rmi-vuln-classloader -p {port} {target}",
        ],
        "credential_targets": ["password", "trust_relationship"],
        "privesc_potential": True,
        "phase": "EXPLOITATION",
    },
    ServiceArchetype.MAIL.value: {
        "attack_priority": 2,
        "reasoning": (
            "MAIL services enable user enumeration and sometimes credential harvesting. "
            "STEP 1: Enumerate users via SMTP VRFY/EXPN/RCPT. STEP 2: Check for open "
            "relay (spam vector, not direct compromise). STEP 3: Look for known MTA "
            "vulnerabilities (Exim, Postfix, Sendmail CVEs). STEP 4: Phishing vector if "
            "social engineering is in scope."
        ),
        "default_tools": [
            "smtp-user-enum -M VRFY -U /usr/share/wordlists/names.txt -t {target}",
            "nmap --script smtp-enum-users -p {port} {target}",
        ],
        "credential_targets": ["email_address", "username"],
        "privesc_potential": False,
        "phase": "ENUMERATION",
    },
    ServiceArchetype.DNS.value: {
        "attack_priority": 2,
        "reasoning": (
            "DNS services enable domain reconnaissance. STEP 1: Attempt zone transfer "
            "(dig axfr). STEP 2: Enumerate subdomains. STEP 3: Check for DNS cache "
            "poisoning vulnerabilities. Usually lower priority unless zone transfer "
            "reveals internal infrastructure."
        ),
        "default_tools": [
            "dig axfr @{target}",
            "dnsrecon -d {target}",
            "nmap --script dns-zone-transfer -p 53 {target}",
        ],
        "credential_targets": [],
        "privesc_potential": False,
        "phase": "RECON",
    },
    ServiceArchetype.DIRECTORY.value: {
        "attack_priority": 5,
        "reasoning": (
            "DIRECTORY services (LDAP/AD) are high-value for domain compromise. "
            "STEP 1: Anonymous LDAP bind to enumerate users, groups, OUs. "
            "STEP 2: AS-REP roasting for accounts without pre-auth. "
            "STEP 3: Kerberoasting for service accounts. STEP 4: Check for "
            "unconstrained delegation. STEP 5: BloodHound enumeration for "
            "attack path discovery."
        ),
        "default_tools": [
            "ldapsearch -x -H ldap://{target} -b '' -s base",
            "enum4linux -a {target}",
            "GetNPUsers.py -dc-ip {target} -no-pass -usersfile users.txt",
            "GetUserSPNs.py -dc-ip {target} -request",
        ],
        "credential_targets": ["password", "ntlm_hash", "kerberos_ticket"],
        "privesc_potential": True,
        "phase": "ENUMERATION",
    },
    ServiceArchetype.MONITORING.value: {
        "attack_priority": 2,
        "reasoning": (
            "MONITORING services (SNMP) can leak system information and sometimes "
            "credentials. STEP 1: Enumerate SNMP with common community strings "
            "(public, private). STEP 2: Walk the MIB tree for system info, "
            "running processes, network config. STEP 3: Check for write access "
            "to change configs."
        ),
        "default_tools": [
            "snmpwalk -v2c -c public {target}",
            "onesixtyone -c /usr/share/wordlists/community.txt {target}",
        ],
        "credential_targets": ["community_string"],
        "privesc_potential": False,
        "phase": "ENUMERATION",
    },
    ServiceArchetype.MESSAGING.value: {
        "attack_priority": 3,
        "reasoning": (
            "MESSAGING services (IRC) sometimes have backdoors or exploitable daemons. "
            "STEP 1: Connect and check daemon version. STEP 2: Check for known backdoors "
            "(UnrealIRCd 3.2.8.1 has a well-known backdoor). STEP 3: Enumerate channels "
            "and users for intelligence gathering."
        ),
        "default_tools": [
            "nmap --script irc-info -p {port} {target}",
            "irssi -c {target} -p {port}",
        ],
        "credential_targets": [],
        "privesc_potential": True,
        "phase": "EXPLOITATION",
    },
    ServiceArchetype.VNC_RDP.value: {
        "attack_priority": 4,
        "reasoning": (
            "VNC/RDP give GUI remote access — direct interactive control if compromised. "
            "STEP 1: Check for no-password or default password VNC (password: 'password'). "
            "STEP 2: For RDP, try common credentials. STEP 3: Check for BlueKeep (CVE-2019-0708) "
            "on older Windows RDP. STEP 4: Hydra brute-force as last resort."
        ),
        "default_tools": [
            "vncviewer {target}:{port}",
            "nmap --script vnc-brute -p {port} {target}",
            "nmap --script rdp-vuln-ms12-020 -p {port} {target}",
        ],
        "credential_targets": ["password", "vnc_password"],
        "privesc_potential": True,
        "phase": "EXPLOITATION",
    },
    ServiceArchetype.CONTAINER.value: {
        "attack_priority": 5,
        "reasoning": (
            "CONTAINER APIs (Docker, Kubernetes) can give full host control. "
            "STEP 1: Check for unauthenticated Docker API (port 2375). "
            "STEP 2: List containers, images. STEP 3: Mount host filesystem "
            "into a new container → read /etc/shadow, plant SSH keys. "
            "STEP 4: For Kubernetes, check for anonymous RBAC access."
        ),
        "default_tools": [
            "curl http://{target}:{port}/version",
            "curl http://{target}:{port}/containers/json",
            "docker -H {target}:{port} ps",
        ],
        "credential_targets": ["api_token", "kubeconfig"],
        "privesc_potential": True,
        "phase": "EXPLOITATION",
    },
    ServiceArchetype.CI_CD.value: {
        "attack_priority": 4,
        "reasoning": (
            "CI/CD services (Jenkins, GitLab) often have weak auth and RCE capabilities. "
            "STEP 1: Check for unauthenticated access. STEP 2: For Jenkins, check "
            "Script Console (/script) for Groovy RCE. STEP 3: Enumerate build jobs "
            "for hardcoded credentials. STEP 4: Check for Git repos with secrets."
        ),
        "default_tools": [
            "curl http://{target}:{port}/",
            "curl http://{target}:{port}/script",
            "nmap --script http-enum -p {port} {target}",
        ],
        "credential_targets": ["password", "api_token", "build_secret"],
        "privesc_potential": True,
        "phase": "ENUMERATION",
    },
    ServiceArchetype.CUSTOM_APP.value: {
        "attack_priority": 3,
        "reasoning": (
            "CUSTOM/UNKNOWN services require careful probing. STEP 1: Banner grab to "
            "identify the service. STEP 2: Connect with nc/telnet and observe protocol. "
            "STEP 3: Fuzz inputs for crashes or unexpected behavior. STEP 4: Search for "
            "the product name + 'exploit' in ExploitDB/CVE databases."
        ),
        "default_tools": [
            "nmap -sV -sC -p {port} {target}",
            "nc -nv {target} {port}",
            "nmap --script banner -p {port} {target}",
        ],
        "credential_targets": [],
        "privesc_potential": False,
        "phase": "ENUMERATION",
    },
}


# ─── Unknown Target Strategy ────────────────────────────────────────────

class UnknownTargetStrategy:
    """
    Chain-of-thought reasoning engine for unclassified targets.
    
    When TargetProfiler cannot match a target to a known type (MS2, MS3, HTB,
    etc.), this strategy analyzes the target's service archetypes to produce:
    
    1. A chain-of-thought reasoning trace explaining the analysis
    2. An ordered list of attack priorities based on exposed services
    3. Concrete tool recommendations per archetype
    4. Risk assessment and estimated difficulty
    
    This is entirely rule-based — no LLM calls. The reasoning is deterministic
    and fast, making it suitable for per-episode use.
    """

    def assess(self, profile: TargetProfile) -> Dict[str, Any]:
        """
        Produce a chain-of-thought assessment for an unknown target.
        
        Args:
            profile: Target profile with services and archetypes classified.
            
        Returns:
            Dict with keys:
              - reasoning_trace: List of reasoning steps (chain-of-thought)
              - attack_priorities: Ordered list of (archetype, priority, reasoning)
              - recommended_tools: Flattened tool list
              - estimated_difficulty: easy/medium/hard/unknown
              - os_inference: Best OS guess with reasoning
              - credential_opportunities: List of credential targets
              - has_privesc_path: Whether any archetype enables privesc
              - summary: One-paragraph tactical summary
        """
        trace: List[str] = []
        priorities: List[Dict[str, Any]] = []
        tools: List[str] = []
        cred_targets: List[str] = []
        has_privesc = False

        archetypes = profile.service_archetypes
        target_ip = profile.target_ip or "{target}"

        # ── Step 1: Observe what we have ─────────────────────────────────
        n_ports = len(profile.open_ports)
        n_services = len(profile.services)
        trace.append(
            f"STEP 1 — OBSERVATION: Target {target_ip} has {n_ports} open ports "
            f"and {n_services} identified services."
        )

        if not archetypes:
            trace.append("  → No service archetypes classified. Cannot reason further.")
            return {
                "reasoning_trace": trace,
                "attack_priorities": [],
                "recommended_tools": ["nmap -sV -sC -A -p- {target}"],
                "estimated_difficulty": "unknown",
                "os_inference": profile.os_family,
                "credential_opportunities": [],
                "has_privesc_path": False,
                "summary": "Insufficient data. Run comprehensive nmap scan first.",
            }

        arch_names = sorted(archetypes.keys())
        trace.append(f"  → Archetypes present: {', '.join(arch_names)}")

        # ── Step 2: Infer OS if unknown ──────────────────────────────────
        os_guess = profile.os_family
        os_reason = ""
        if os_guess == "unknown":
            os_guess, os_reason = self._infer_os(profile)
            trace.append(f"STEP 2 — OS INFERENCE: {os_reason}")
        else:
            trace.append(f"STEP 2 — OS KNOWN: {os_guess}")

        # ── Step 3: Prioritize archetypes ────────────────────────────────
        trace.append("STEP 3 — PRIORITIZATION: Ranking attack surfaces by value.")

        for arch_name, ports in archetypes.items():
            meta = _ARCHETYPE_REASONING.get(arch_name, _ARCHETYPE_REASONING["custom_app"])
            priority = meta["attack_priority"]
            reasoning = meta["reasoning"]

            # Boost priority if multiple services of same type
            if len(ports) > 1:
                priority = min(priority + 1, 5)
                reasoning_note = f" [BOOSTED: {len(ports)} services of this type]"
            else:
                reasoning_note = ""

            priorities.append({
                "archetype": arch_name,
                "ports": ports,
                "priority": priority,
                "reasoning": reasoning + reasoning_note,
                "phase": meta["phase"],
            })

            # Collect tools with port/target substitution
            for tool_template in meta["default_tools"]:
                port = ports[0]  # Primary port for this archetype
                tool = tool_template.format(
                    target=target_ip, port=port,
                    proto=arch_name.split("_")[0] if "_" in arch_name else arch_name,
                )
                tools.append(tool)

            # Collect credential targets
            cred_targets.extend(meta.get("credential_targets", []))

            if meta.get("privesc_potential"):
                has_privesc = True

            trace.append(
                f"  → {arch_name} (ports {ports}): priority={priority}, "
                f"phase={meta['phase']}{reasoning_note}"
            )

        # Sort by priority descending
        priorities.sort(key=lambda x: x["priority"], reverse=True)

        # ── Step 4: Estimate difficulty ──────────────────────────────────
        difficulty = self._estimate_difficulty(profile, archetypes, has_privesc)
        trace.append(f"STEP 4 — DIFFICULTY: Estimated as '{difficulty}'.")

        # ── Step 5: Build tactical summary ───────────────────────────────
        top_archetypes = [p["archetype"] for p in priorities[:3]]
        top_ports = []
        for p in priorities[:3]:
            top_ports.extend(p["ports"][:2])

        summary_parts = [
            f"Unknown target with {n_ports} open ports on {os_guess}.",
            f"Top attack surfaces: {', '.join(top_archetypes)}.",
        ]
        if has_privesc:
            summary_parts.append("Privilege escalation paths available.")
        if ServiceArchetype.REMOTE_EXEC.value in archetypes:
            summary_parts.append("CRITICAL: Remote exec services detected — check for instant shells.")
        if ServiceArchetype.DATABASE.value in archetypes:
            summary_parts.append("Databases found — try default credentials for quick wins.")
        summary_parts.append(f"Estimated difficulty: {difficulty}.")

        summary = " ".join(summary_parts)
        trace.append(f"STEP 5 — SUMMARY: {summary}")

        # De-duplicate
        cred_targets = list(dict.fromkeys(cred_targets))

        return {
            "reasoning_trace": trace,
            "attack_priorities": priorities,
            "recommended_tools": tools[:20],  # Cap at 20 tools
            "estimated_difficulty": difficulty,
            "os_inference": os_guess,
            "os_inference_reason": os_reason,
            "credential_opportunities": cred_targets,
            "has_privesc_path": has_privesc,
            "summary": summary,
        }

    def _infer_os(self, profile: TargetProfile) -> Tuple[str, str]:
        """Infer OS from service fingerprints when OS is unknown."""
        linux_signals = 0
        windows_signals = 0
        reasons: List[str] = []

        for svc in profile.services:
            product_lower = (svc.product or "").lower()
            service_lower = (svc.service or "").lower()
            combined = f"{product_lower} {service_lower}"

            if any(x in combined for x in ("openssh", "vsftpd", "proftpd", "apache",
                                            "nginx", "pure-ftpd", "postfix", "exim")):
                linux_signals += 1
                reasons.append(f"{svc.port}: {svc.product or svc.service} → linux")
            elif any(x in combined for x in ("iis", "microsoft", "ms-wbt", "ms-sql",
                                              "windows", "rdp", "netbios")):
                windows_signals += 1
                reasons.append(f"{svc.port}: {svc.product or svc.service} → windows")

        # Port-based heuristics
        if profile.open_ports & {512, 513, 514, 2049}:
            linux_signals += 1
            reasons.append("rexec/rlogin/rsh/NFS → linux")
        if profile.open_ports & {135, 139, 445, 3389}:
            windows_signals += 1
            reasons.append("SMB/RDP → windows")

        if linux_signals > windows_signals:
            reason = f"Inferred LINUX ({linux_signals} signals vs {windows_signals} windows): {'; '.join(reasons[:3])}"
            return "linux", reason
        elif windows_signals > linux_signals:
            reason = f"Inferred WINDOWS ({windows_signals} signals vs {linux_signals} linux): {'; '.join(reasons[:3])}"
            return "windows", reason
        else:
            return "unknown", f"Could not determine OS ({linux_signals} linux, {windows_signals} windows signals)"

    def _estimate_difficulty(
        self,
        profile: TargetProfile,
        archetypes: Dict[str, List[int]],
        has_privesc: bool,
    ) -> str:
        """Estimate target difficulty based on exposed services."""
        easy_indicators = 0
        hard_indicators = 0

        # Easy indicators
        if ServiceArchetype.REMOTE_EXEC.value in archetypes:
            easy_indicators += 2  # Direct exec = likely easy
        if len(profile.open_ports) > 10:
            easy_indicators += 1  # Many ports = larger attack surface
        if ServiceArchetype.DATABASE.value in archetypes:
            easy_indicators += 1  # DBs often have default creds
        if 1524 in profile.open_ports:
            easy_indicators += 3  # Known backdoor port

        # Hard indicators
        if len(profile.open_ports) <= 2:
            hard_indicators += 2  # Minimal surface
        if not has_privesc:
            hard_indicators += 1
        web_only = (
            set(archetypes.keys()) <= {ServiceArchetype.WEB.value, ServiceArchetype.CUSTOM_APP.value}
        )
        if web_only:
            hard_indicators += 1  # Web-only usually needs more skill

        if easy_indicators >= 3:
            return "easy"
        elif hard_indicators >= 3:
            return "hard"
        elif easy_indicators > hard_indicators:
            return "medium"
        elif hard_indicators > easy_indicators:
            return "hard"
        return "medium"


# ─── Generalization Playbook ─────────────────────────────────────────────

@dataclass
class GeneralizationPlaybook:
    """
    Fallback multi-phase attack playbook built from service archetypes.
    
    When no known target-specific playbook exists, this generates a
    generic but structured attack plan based purely on what services
    were observed. Each chain targets a specific archetype with ordered steps.
    """
    target_ip: str
    os_family: str
    chains: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""

    @classmethod
    def from_profile(cls, profile: TargetProfile) -> "GeneralizationPlaybook":
        """
        Build a generalization playbook from a target profile.
        
        Creates attack chains ordered by archetype priority, each with
        concrete steps adapted to the target's actual ports and services.
        """
        playbook = cls(
            target_ip=profile.target_ip or "{target}",
            os_family=profile.os_family,
        )

        if not profile.service_archetypes:
            playbook.reasoning = "No archetypes classified — run nmap -sV first."
            playbook.chains.append({
                "name": "Initial Reconnaissance",
                "summary": "Comprehensive port and service scan",
                "steps": [
                    {"step": 1, "command": f"nmap -sV -sC -A -p- {playbook.target_ip}"},
                    {"step": 2, "command": f"nmap -sU --top-ports 100 {playbook.target_ip}"},
                ],
                "source": "generalization",
            })
            return playbook

        # Sort archetypes by attack priority
        sorted_archetypes = []
        for arch_name, ports in profile.service_archetypes.items():
            meta = _ARCHETYPE_REASONING.get(arch_name, _ARCHETYPE_REASONING["custom_app"])
            sorted_archetypes.append((meta["attack_priority"], arch_name, ports, meta))
        sorted_archetypes.sort(key=lambda x: x[0], reverse=True)

        reasoning_parts = [
            f"Generated playbook for unknown {profile.os_family} target "
            f"with {len(profile.open_ports)} open ports.",
        ]

        # Build chains per archetype (top 5)
        for priority, arch_name, ports, meta in sorted_archetypes[:5]:
            chain_steps = []
            step_num = 1

            # Always start with detailed port scan
            for port in ports[:3]:  # Max 3 ports per archetype chain
                chain_steps.append({
                    "step": step_num,
                    "command": f"nmap -sV -sC -p {port} {playbook.target_ip}",
                    "purpose": "Detailed service enumeration",
                })
                step_num += 1

            # Add archetype-specific tools
            for tool_template in meta["default_tools"][:3]:
                port = ports[0]
                proto = arch_name.split("_")[0] if "_" in arch_name else arch_name
                cmd = tool_template.format(
                    target=playbook.target_ip, port=port, proto=proto,
                )
                chain_steps.append({
                    "step": step_num,
                    "command": cmd,
                    "purpose": f"{arch_name} attack technique",
                })
                step_num += 1

            # Add privesc step if applicable
            if meta.get("privesc_potential") and profile.os_family == "linux":
                chain_steps.append({
                    "step": step_num,
                    "command": "id && uname -a && sudo -l",
                    "purpose": "Post-exploitation: check privileges",
                })
            elif meta.get("privesc_potential") and profile.os_family == "windows":
                chain_steps.append({
                    "step": step_num,
                    "command": "whoami /priv && systeminfo",
                    "purpose": "Post-exploitation: check privileges",
                })

            chain = {
                "name": f"{arch_name.upper()} Attack Chain",
                "summary": meta["reasoning"][:150],
                "priority": priority,
                "target_ports": ports,
                "steps": chain_steps,
                "source": "generalization",
                "phase": meta["phase"],
            }
            playbook.chains.append(chain)

            reasoning_parts.append(
                f"Chain {len(playbook.chains)}: {arch_name} (priority {priority}, "
                f"ports {ports[:3]})"
            )

        # Add a credential harvesting chain if multiple auth archetypes
        auth_archetypes = [
            a for a in sorted_archetypes
            if a[3].get("credential_targets")
        ]
        if len(auth_archetypes) >= 2:
            cred_chain_steps = []
            step_num = 1
            for _, arch_name, ports, meta in auth_archetypes[:4]:
                for cred_type in meta["credential_targets"][:2]:
                    cred_chain_steps.append({
                        "step": step_num,
                        "command": meta["default_tools"][0].format(
                            target=playbook.target_ip,
                            port=ports[0],
                            proto=arch_name.split("_")[0],
                        ),
                        "purpose": f"Harvest {cred_type} from {arch_name}",
                    })
                    step_num += 1
                    if step_num > 8:
                        break

            playbook.chains.append({
                "name": "Cross-Service Credential Harvesting",
                "summary": "Try default/weak credentials across all auth-capable services",
                "priority": 4,
                "steps": cred_chain_steps,
                "source": "generalization",
                "phase": "EXPLOITATION",
            })
            reasoning_parts.append(
                f"Added credential harvesting chain across {len(auth_archetypes)} "
                "auth-capable archetypes."
            )

        playbook.reasoning = " ".join(reasoning_parts)
        return playbook

    def to_prompt_fragment(self, max_chars: int = 1500) -> str:
        """Convert playbook to a prompt-injectable string."""
        lines = [
            f"=== GENERALIZATION PLAYBOOK ({self.os_family}) ===",
            self.reasoning[:200],
            "",
        ]
        for i, chain in enumerate(self.chains[:5], 1):
            lines.append(f"Chain {i}: {chain['name']} (priority={chain.get('priority', '?')})")
            lines.append(f"  Summary: {chain['summary'][:100]}")
            for step in chain.get("steps", [])[:4]:
                lines.append(f"  [{step['step']}] {step['command']}")
            lines.append("")

        result = "\n".join(lines)
        return result[:max_chars]