"""
Target Profiler — Dynamic Target Intelligence for Ariaska_RL.

Phase 9.2: Generalizes beyond MS2/MS3 to support arbitrary targets
(HackTheBox, CTF, cloud, AD environments) by profiling scan results
and dynamically selecting relevant knowledge packs.

Architecture:
  - TargetProfile: Structured representation of a target's attack surface
  - TargetProfiler: Builds profiles from nmap/scan output and KG data
  - KnowledgeSelector: Picks relevant knowledge subsets for the profiled target

Usage:
    from core.knowledge.target_profiler import TargetProfiler
    profiler = TargetProfiler(knowledge_graph=kg, retriever=kr)
    profile = profiler.profile_from_nmap(nmap_output)
    knowledge = profiler.select_knowledge(profile)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
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
        if self.kr:
            self._enrich_from_retriever(profile)
        if self.kg:
            self._enrich_from_kg(profile)
        self._build_attack_chains(profile)

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
        """
        knowledge = {
            "target_type": profile.target_type,
            "os_family": profile.os_family,
            "relevant_techniques": [],
            "relevant_exploits": [],
            "relevant_cves": profile.matched_cves[:20],
            "attack_chains": profile.attack_chains[:5],
            "privesc_techniques": [],
            "credential_targets": [],
        }
        
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
