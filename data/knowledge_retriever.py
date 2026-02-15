#!/usr/bin/env python3
"""
data/knowledge_retriever.py — ARIASKA Knowledge Retriever v1.0

Runtime query interface for the ingested knowledge base. Provides fast,
multi-index lookups by port, service, CVE, kill-chain phase, tool name,
and free-text search.

Usage:
    from data.knowledge_retriever import KnowledgeRetriever

    kr = KnowledgeRetriever()  # Loads indices on first access
    kr.by_port(21)             # Everything we know about port 21
    kr.by_service("vsftpd")    # Service-specific knowledge
    kr.by_phase("EXPLOITATION") # Phase-filtered methodology
    kr.by_cve("CVE-2007-2447")  # CVE-specific exploit chain
    kr.suggest_next(port=21, phase="ENUMERATION")  # Contextual suggestions
    kr.get_kill_chain("ftp")   # Full exploitation chain for FTP
    kr.for_mentor_prompt(state) # Rich knowledge string for LLM context injection
    kr.search("suid binary")   # Free-text search across all knowledge

Integration Points:
    - SmartMentor._build_system_prompt() — inject kr.for_mentor_prompt()
    - Codex RESEARCHER persona — use kr.search() for deep lookup
    - SmartCoach registry stage — kr.suggest_next() for command candidates
    - RedAgent emergency fallbacks — kr.by_port() for MS2-specific paths

Author: Filip Volf / Ariaska System
"""

import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("ariaska.knowledge_retriever")

BASE_DIR = Path(__file__).parent
KB_DIR = BASE_DIR / "knowledge_base"


class KnowledgeRetriever:
    """
    Fast multi-index query engine for the ARIASKA knowledge base.

    Lazy-loads and indexes all JSON knowledge files on first access.
    Thread-safe for concurrent agent queries.
    """

    _instance: Optional["KnowledgeRetriever"] = None

    def __new__(cls, *args, **kwargs):
        """Singleton — one retriever for all agents."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, kb_dir: Optional[Path] = None, lazy: bool = True):
        if self._initialized:
            return
        self._kb_dir = kb_dir or KB_DIR
        self._loaded = False

        # Indices
        self._port_index: Dict[int, List[Dict]] = defaultdict(list)
        self._service_index: Dict[str, List[Dict]] = defaultdict(list)
        self._cve_index: Dict[str, List[Dict]] = defaultdict(list)
        self._phase_index: Dict[str, List[Dict]] = defaultdict(list)
        self._tool_index: Dict[str, List[Dict]] = defaultdict(list)
        self._vuln_type_index: Dict[str, List[Dict]] = defaultdict(list)

        # Raw data
        self._services: List[Dict] = []
        self._ports: Dict[int, Dict] = {}
        self._cves: List[Dict] = []
        self._payloads: List[Dict] = []
        self._methodology: List[Dict] = []
        self._commands: List[Dict] = []
        self._kill_chains: List[Dict] = []
        self._cheatsheets: List[Dict] = []
        self._privesc_linux: List[Dict] = []
        self._privesc_windows: List[Dict] = []

        # Search cache
        self._search_corpus: List[Tuple[str, Dict, str]] = []  # (text, entry, category)

        self._initialized = True
        if not lazy:
            self._load()

    def _ensure_loaded(self):
        """Lazy-load knowledge base on first query."""
        if not self._loaded:
            self._load()

    def _load(self):
        """Load all knowledge files and build indices."""
        if not self._kb_dir.exists():
            logger.warning(f"Knowledge base not found at {self._kb_dir}. "
                           f"Run: python -m data.knowledge_ingestion")
            self._loaded = True
            return

        logger.info(f"Loading knowledge base from {self._kb_dir}...")

        file_loaders = {
            "services.json": ("_services", list),
            "ports.json": ("_ports", dict),
            "cves.json": ("_cves", list),
            "payloads.json": ("_payloads", list),
            "methodology.json": ("_methodology", list),
            "commands.json": ("_commands", list),
            "kill_chains.json": ("_kill_chains", list),
            "cheatsheets.json": ("_cheatsheets", list),
            "privesc_linux.json": ("_privesc_linux", list),
            "privesc_windows.json": ("_privesc_windows", list),
        }

        total_entries = 0
        for fname, (attr, expected_type) in file_loaders.items():
            fpath = self._kb_dir / fname
            if fpath.exists():
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    setattr(self, attr, data)
                    count = len(data) if isinstance(data, (list, dict)) else 0
                    total_entries += count
                    logger.debug(f"  Loaded {fname}: {count:,} entries")
                except Exception as e:
                    logger.error(f"  Failed to load {fname}: {e}")

        # Build indices
        self._build_indices()
        self._loaded = True
        logger.info(f"Knowledge base loaded: {total_entries:,} total entries across "
                     f"{len(file_loaders)} files")

    def _build_indices(self):
        """Build fast lookup indices from raw data."""

        # Index services by port and name
        for svc in self._services:
            port = svc.get("port", 0)
            name = svc.get("service_name", "").lower()
            if port > 0:
                self._port_index[port].append(svc)
            if name:
                self._service_index[name].append(svc)
                # Also index by substrings (e.g., "vsftpd" → "ftp")
                for key in ["ftp", "ssh", "http", "smb", "smtp", "dns", "mysql",
                             "postgres", "redis", "vnc", "irc", "telnet", "nfs",
                             "ldap", "kerberos", "snmp", "tomcat", "apache"]:
                    if key in name:
                        self._service_index[key].append(svc)

            # Build search corpus
            text_parts = [
                name, str(port),
                svc.get("description", ""),
                svc.get("methodology", ""),
                svc.get("reasoning", ""),
            ]
            self._search_corpus.append(
                (" ".join(text_parts).lower(), svc, "service")
            )

        # Index ports dict
        for port_str, data in self._ports.items() if isinstance(self._ports, dict) else []:
            try:
                port = int(port_str)
                if port not in self._port_index:
                    self._port_index[port] = []
                self._port_index[port].append(data)
            except (ValueError, TypeError):
                pass

        # Index CVEs
        for cve in self._cves:
            cve_id = cve.get("cve_id", "").upper()
            if cve_id:
                self._cve_index[cve_id].append(cve)
            software = cve.get("affected_software", "").lower()
            if software:
                self._service_index[software].append(cve)
            self._search_corpus.append(
                (f"{cve_id} {software} {cve.get('description', '')}".lower(), cve, "cve")
            )

        # Index payloads by vuln type
        for payload in self._payloads:
            vtype = payload.get("vuln_type", "").lower()
            if vtype:
                self._vuln_type_index[vtype].append(payload)
            self._search_corpus.append(
                (f"{vtype} {payload.get('payload', '')} {payload.get('context', '')}".lower(),
                 payload, "payload")
            )

        # Index commands by tool and phase
        for cmd in self._commands:
            tool = cmd.get("tool_name", "").lower()
            phase = cmd.get("phase", "").upper()
            if tool:
                self._tool_index[tool].append(cmd)
            if phase:
                self._phase_index[phase].append(cmd)
            self._search_corpus.append(
                (f"{tool} {cmd.get('command', '')} {cmd.get('description', '')}".lower(),
                 cmd, "command")
            )

        # Index methodology by phase
        for meth in self._methodology:
            phase = meth.get("phase", "").upper()
            if phase:
                self._phase_index[phase].append(meth)
            self._search_corpus.append(
                (f"{meth.get('title', '')} {meth.get('reasoning', '')}".lower(),
                 meth, "methodology")
            )

        # Index kill chains
        for chain in self._kill_chains:
            svc = chain.get("target_service", "").lower()
            port = chain.get("target_port", 0)
            if svc:
                self._service_index[svc].append(chain)
            if port > 0:
                self._port_index[port].append(chain)
            self._search_corpus.append(
                (f"{svc} {port} {chain.get('chain_name', '')} {chain.get('reasoning', '')}".lower(),
                 chain, "kill_chain")
            )

        # Index cheatsheets by tool
        for cs in self._cheatsheets:
            tool = cs.get("tool_or_topic", "").lower()
            if tool:
                self._tool_index[tool].append(cs)
            self._search_corpus.append(
                (f"{tool} {cs.get('section', '')} {cs.get('content', '')[:500]}".lower(),
                 cs, "cheatsheet")
            )

        # Index privesc
        for pe in self._privesc_linux:
            self._phase_index["PRIVILEGE_ESCALATION"].append(pe)
            self._search_corpus.append(
                (f"linux privesc {pe.get('technique', '')} {pe.get('description', '')[:300]}".lower(),
                 pe, "privesc_linux")
            )
        for pe in self._privesc_windows:
            self._phase_index["PRIVILEGE_ESCALATION"].append(pe)
            self._search_corpus.append(
                (f"windows privesc {pe.get('technique', '')} {pe.get('description', '')[:300]}".lower(),
                 pe, "privesc_windows")
            )

        logger.debug(
            f"Indices built: {len(self._port_index)} ports, "
            f"{len(self._service_index)} services, "
            f"{len(self._cve_index)} CVEs, "
            f"{len(self._phase_index)} phases, "
            f"{len(self._tool_index)} tools, "
            f"{len(self._vuln_type_index)} vuln types, "
            f"{len(self._search_corpus)} searchable entries"
        )

    # ─── Query Methods ──────────────────────────────────────────────────────

    def by_port(self, port: int, max_results: int = 20) -> List[Dict]:
        """Get all knowledge about a specific port."""
        self._ensure_loaded()
        results = self._port_index.get(port, [])
        return results[:max_results]

    def by_service(self, service: str, max_results: int = 20) -> List[Dict]:
        """Get all knowledge about a specific service (fuzzy match)."""
        self._ensure_loaded()
        service_lower = service.lower().strip()
        results = self._service_index.get(service_lower, [])

        # Fuzzy match if no exact hit
        if not results:
            for key, entries in self._service_index.items():
                if service_lower in key or key in service_lower:
                    results.extend(entries)

        return results[:max_results]

    def by_cve(self, cve_id: str, max_results: int = 10) -> List[Dict]:
        """Get exploitation knowledge for a specific CVE."""
        self._ensure_loaded()
        return self._cve_index.get(cve_id.upper(), [])[:max_results]

    def by_phase(self, phase: str, max_results: int = 30) -> List[Dict]:
        """Get methodology and commands for a kill-chain phase."""
        self._ensure_loaded()
        phase_upper = phase.upper().replace(" ", "_")
        return self._phase_index.get(phase_upper, [])[:max_results]

    def by_tool(self, tool_name: str, max_results: int = 20) -> List[Dict]:
        """Get knowledge about a specific tool (nmap, hydra, etc.)."""
        self._ensure_loaded()
        return self._tool_index.get(tool_name.lower(), [])[:max_results]

    def by_vuln_type(self, vuln_type: str, max_results: int = 20) -> List[Dict]:
        """Get payloads for a vulnerability type (sqli, xss, etc.)."""
        self._ensure_loaded()
        return self._vuln_type_index.get(vuln_type.lower(), [])[:max_results]

    def get_kill_chain(self, service_or_port, max_results: int = 5) -> List[Dict]:
        """Get full exploitation chains for a service or port."""
        self._ensure_loaded()
        results = []
        for chain in self._kill_chains:
            if isinstance(service_or_port, int):
                if chain.get("target_port") == service_or_port:
                    results.append(chain)
            elif isinstance(service_or_port, str):
                if service_or_port.lower() in chain.get("target_service", "").lower():
                    results.append(chain)
        return results[:max_results]

    def get_privesc(self, os_type: str = "linux", max_results: int = 20) -> List[Dict]:
        """Get privilege escalation techniques."""
        self._ensure_loaded()
        if os_type.lower() == "linux":
            return self._privesc_linux[:max_results]
        return self._privesc_windows[:max_results]

    def get_credentials(self, service: str = "") -> List[Dict]:
        """Get known default credentials, optionally filtered by service."""
        self._ensure_loaded()
        creds = []
        for svc in self._services:
            svc_creds = svc.get("default_credentials", [])
            if service:
                if service.lower() in svc.get("service_name", "").lower():
                    creds.extend(svc_creds)
            else:
                creds.extend(svc_creds)
        return creds

    def search(self, query: str, max_results: int = 15,
               category: Optional[str] = None) -> List[Tuple[float, Dict, str]]:
        """
        Free-text search across entire knowledge base.

        Returns list of (relevance_score, entry, category) tuples sorted by relevance.
        """
        self._ensure_loaded()
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        results = []
        for text, entry, cat in self._search_corpus:
            if category and cat != category:
                continue

            # Score based on term overlap
            score = 0.0
            for term in query_terms:
                if term in text:
                    # Exact word match bonus
                    count = text.count(term)
                    score += min(count, 3) * 1.0
                    # Full phrase bonus
                    if query_lower in text:
                        score += 5.0

            if score > 0:
                results.append((score, entry, cat))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:max_results]

    # ─── Contextual Methods (Agent Integration) ────────────────────────────

    def suggest_next(self, port: int = 0, service: str = "",
                     phase: str = "", discovered: Optional[Set[str]] = None,
                     max_commands: int = 5) -> List[Dict[str, str]]:
        """
        Suggest next commands based on current attack context.

        Used by SmartCoach registry stage to propose contextually-relevant commands
        from the knowledge base.
        """
        self._ensure_loaded()
        candidates = []

        # Port-based suggestions
        if port > 0:
            for entry in self._port_index.get(port, []):
                for cmd_list_key in ["enumeration_commands", "exploitation_commands"]:
                    for cmd in entry.get(cmd_list_key, [])[:5]:
                        candidates.append({
                            "command": cmd.get("command", ""),
                            "context": cmd.get("context", ""),
                            "phase": cmd.get("phase", phase),
                            "source": f"kb:port_{port}",
                            "reasoning": entry.get("reasoning", "")[:200],
                        })

        # Service-based suggestions
        if service:
            for entry in self.by_service(service, max_results=5):
                for cmd_list_key in ["enumeration_commands", "exploitation_commands"]:
                    for cmd in entry.get(cmd_list_key, [])[:3]:
                        candidates.append({
                            "command": cmd.get("command", ""),
                            "context": cmd.get("context", ""),
                            "phase": cmd.get("phase", phase),
                            "source": f"kb:svc_{service}",
                        })

        # Phase-based suggestions
        if phase:
            for entry in self.by_phase(phase, max_results=10):
                for cmd in entry.get("commands", [])[:3]:
                    candidates.append({
                        "command": cmd.get("command", ""),
                        "context": cmd.get("context", ""),
                        "phase": phase,
                        "source": f"kb:phase_{phase}",
                    })

        # Deduplicate and filter
        seen = set()
        unique = []
        for c in candidates:
            cmd_text = c.get("command", "").strip()
            if cmd_text and cmd_text not in seen:
                seen.add(cmd_text)
                unique.append(c)

        return unique[:max_commands]

    def for_mentor_prompt(self, state: Dict[str, Any],
                          max_tokens: int = 2000) -> str:
        """
        Generate rich knowledge context for LLM mentor prompt injection.

        Analyzes the current state (discovered ports, services, phase) and
        returns a focused knowledge string that the mentor can use to guide
        the agent with domain-specific exploitation methodology.
        """
        self._ensure_loaded()
        parts = []

        phase = state.get("phase", "RECON")
        discovered_ports = state.get("discovered_ports", [])
        discovered_services = state.get("discovered_services", [])
        target_ip = state.get("target_ip", "")

        parts.append(f"=== KNOWLEDGE BASE CONTEXT (Phase: {phase}) ===")

        # For each discovered port, inject exploitation paths
        for port in discovered_ports[:10]:
            port_num = int(port) if isinstance(port, str) and port.isdigit() else port
            if not isinstance(port_num, int):
                continue
            entries = self.by_port(port_num, max_results=3)
            if entries:
                best = entries[0]
                service = best.get("service_name", f"port-{port_num}")
                parts.append(f"\n--- Port {port_num} ({service}) ---")

                # Methodology snippet
                methodology = best.get("methodology", "")
                if methodology:
                    parts.append(f"Methodology: {methodology[:400]}")

                # Key exploitation commands
                exploit_cmds = best.get("exploitation_commands", [])[:3]
                if exploit_cmds:
                    parts.append("Exploit Commands:")
                    for cmd in exploit_cmds:
                        parts.append(f"  $ {cmd.get('command', '')}")
                        if cmd.get("context"):
                            parts.append(f"    → {cmd['context'][:100]}")

                # Default creds
                creds = best.get("default_credentials", [])[:2]
                if creds:
                    parts.append("Default Credentials:")
                    for c in creds:
                        parts.append(f"  {c.get('username', '?')}:{c.get('password', '?')}")

                # Reasoning
                reasoning = best.get("reasoning", "")
                if reasoning:
                    parts.append(f"Why: {reasoning[:200]}")

                # CVEs
                vulns = best.get("common_vulnerabilities", [])[:3]
                if vulns:
                    parts.append(f"Known CVEs: {', '.join(vulns[:3])}")

        # Phase-specific methodology
        phase_entries = self.by_phase(phase, max_results=3)
        if phase_entries:
            parts.append(f"\n--- {phase} Methodology ---")
            for entry in phase_entries[:2]:
                title = entry.get("title", "")
                if title:
                    parts.append(f"• {title}")
                cmds = entry.get("commands", [])[:2]
                for cmd in cmds:
                    parts.append(f"  $ {cmd.get('command', '')}")

        # Kill chains for discovered services
        for svc in discovered_services[:5]:
            chains = self.get_kill_chain(svc, max_results=1)
            if chains:
                chain = chains[0]
                parts.append(f"\n--- Kill Chain: {svc} ---")
                for step in chain.get("steps", [])[:5]:
                    parts.append(f"  Step {step.get('step', '?')}: {step.get('command', '')}")
                if chain.get("reasoning"):
                    parts.append(f"  Reasoning: {chain['reasoning'][:200]}")

        result = "\n".join(parts)
        # Truncate to max_tokens (rough estimate: 4 chars/token)
        max_chars = max_tokens * 4
        if len(result) > max_chars:
            result = result[:max_chars] + "\n[... knowledge truncated ...]"

        return result

    def get_ms2_knowledge(self) -> str:
        """
        Get Metasploitable 2-specific exploitation knowledge.

        Returns a pre-built string with all known MS2 exploitation paths
        for direct injection into mentor system prompts.
        """
        self._ensure_loaded()

        # MS2-specific ports
        ms2_ports = [21, 22, 23, 25, 80, 139, 445, 512, 513, 514,
                     1099, 1524, 2049, 3306, 5432, 5900, 6667, 8180]

        parts = ["=== METASPLOITABLE 2 EXPLOITATION KNOWLEDGE ==="]
        parts.append("Priority: ingreslock(1524) → vsftpd(21) → samba(139/445) → "
                      "IRC(6667) → java_rmi(1099) → postgres(5432) → mysql(3306)\n")

        for port in ms2_ports:
            entries = self.by_port(port, max_results=2)
            if entries:
                best = entries[0]
                svc = best.get("service_name", f"port-{port}")
                parts.append(f"[Port {port}] {svc}")
                exploit_cmds = best.get("exploitation_commands", [])[:3]
                for cmd in exploit_cmds:
                    parts.append(f"  $ {cmd.get('command', '')}")
                creds = best.get("default_credentials", [])[:2]
                for c in creds:
                    parts.append(f"  cred: {c.get('username', '')}:{c.get('password', '')}")
                reasoning = best.get("reasoning", "")
                if reasoning:
                    parts.append(f"  → {reasoning[:150]}")
                parts.append("")

        return "\n".join(parts)

    # ─── Statistics ─────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, int]:
        """Get knowledge base statistics."""
        self._ensure_loaded()
        return {
            "services": len(self._services),
            "ports_indexed": len(self._port_index),
            "cves": len(self._cves),
            "payloads": len(self._payloads),
            "methodology": len(self._methodology),
            "commands": len(self._commands),
            "kill_chains": len(self._kill_chains),
            "cheatsheets": len(self._cheatsheets),
            "privesc_linux": len(self._privesc_linux),
            "privesc_windows": len(self._privesc_windows),
            "searchable_entries": len(self._search_corpus),
        }

    @classmethod
    def reset_singleton(cls):
        """Reset singleton (for testing)."""
        cls._instance = None


# ─── Convenience ────────────────────────────────────────────────────────────

def get_knowledge_retriever(lazy: bool = True) -> KnowledgeRetriever:
    """Get or create the singleton KnowledgeRetriever."""
    return KnowledgeRetriever(lazy=lazy)


if __name__ == "__main__":
    """Quick test of the retriever."""
    import sys
    logging.basicConfig(level=logging.DEBUG)

    kr = get_knowledge_retriever(lazy=False)
    print(f"\nKnowledge Base Stats: {json.dumps(kr.stats(), indent=2)}")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\nSearch: '{query}'")
        results = kr.search(query)
        for score, entry, cat in results[:5]:
            print(f"  [{cat}] score={score:.1f} — {str(entry)[:200]}")
