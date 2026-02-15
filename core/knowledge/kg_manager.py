"""
Ariaska Knowledge Graph (AKG) — LMDB-backed Graph with NetworkX in-memory view.

Phase 9.2: Core knowledge graph infrastructure for Ariaska_RL.

Node types:
  - Host          — IP/hostname with discovered info
  - Service       — port/service detected on a host
  - CVE           — vulnerability with severity and exploit info
  - Technique     — MITRE ATT&CK technique (T-codes)
  - Binary        — GTFOBins/LOLBAS binary abuse method
  - Exploit       — ExploitDB or Metasploit module
  - Credential    — discovered credential pair
  - Tool          — pentesting tool (nmap, hydra, msfconsole, etc.)
  - Phase         — attack phase node (RECON, EXPLOITATION, etc.)

Edge types:
  - HOSTS         — Host → Service
  - EXPLOITS      — Exploit/Technique → CVE/Service
  - USES_TOOL     — Technique → Tool
  - REQUIRES      — Technique → Technique (prerequisite)
  - ESCALATES_TO  — lower priv → higher priv
  - FOUND_ON      — CVE → Service
  - CREDENTIALS_FOR — Credential → Service
  - PHASE_OF      — Technique/Tool → Phase
  - SIMILAR_TO    — semantic similarity edge (embedding-based)

Storage:
  - LMDB for persistent key-value store (node/edge serialization)
  - NetworkX DiGraph for in-memory traversal, shortest-path, subgraph queries
  - Optional: sentence-transformer embeddings for semantic similarity

Usage:
    from core.knowledge.kg_manager import KnowledgeGraph
    kg = KnowledgeGraph()
    kg.add_node("CVE-2007-2447", node_type="CVE", severity="critical",
                description="Samba usermap_script RCE")
    kg.add_edge("CVE-2007-2447", "samba:139", edge_type="FOUND_ON")
    exploits = kg.get_exploits_for_service("samba", 139)
    path = kg.shortest_attack_path("recon", "root_shell")
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("ariaska.kg_manager")


# ─── Enums ──────────────────────────────────────────────────────────────────

class NodeType(str, Enum):
    HOST = "host"
    SERVICE = "service"
    CVE = "cve"
    TECHNIQUE = "technique"
    BINARY = "binary"
    EXPLOIT = "exploit"
    CREDENTIAL = "credential"
    TOOL = "tool"
    PHASE = "phase"
    METHODOLOGY = "methodology"


class EdgeType(str, Enum):
    HOSTS = "hosts"
    EXPLOITS = "exploits"
    USES_TOOL = "uses_tool"
    REQUIRES = "requires"
    ESCALATES_TO = "escalates_to"
    FOUND_ON = "found_on"
    CREDENTIALS_FOR = "credentials_for"
    PHASE_OF = "phase_of"
    SIMILAR_TO = "similar_to"
    LEADS_TO = "leads_to"
    PART_OF = "part_of"
    MITIGATED_BY = "mitigated_by"


# ─── Node / Edge dataclasses ────────────────────────────────────────────────

@dataclass
class KGNode:
    """A node in the knowledge graph."""
    node_id: str
    node_type: NodeType
    label: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None  # sentence-transformer vector
    created_at: float = field(default_factory=time.time)
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["node_type"] = self.node_type.value
        if self.embedding:
            d.pop("embedding", None)  # Don't serialize embeddings to JSON
        return d


@dataclass
class KGEdge:
    """A directed edge in the knowledge graph."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["edge_type"] = self.edge_type.value
        return d


# ─── Knowledge Graph ────────────────────────────────────────────────────────

class KnowledgeGraph:
    """
    LMDB-backed knowledge graph with NetworkX in-memory traversal.

    - Persistent storage in LMDB for durability across restarts
    - NetworkX DiGraph for fast in-memory queries (shortest path, neighbors, etc.)
    - Optional sentence-transformer embeddings for semantic search
    - Bulk loading from ingested JSON knowledge base
    """

    _instance: Optional["KnowledgeGraph"] = None

    @classmethod
    def get_instance(cls) -> Optional["KnowledgeGraph"]:
        """Get the singleton KnowledgeGraph instance (set by SmartOrchestrator)."""
        return cls._instance

    @classmethod
    def set_instance(cls, instance: "KnowledgeGraph"):
        """Set the singleton instance."""
        cls._instance = instance

    def __init__(
        self,
        db_path: Optional[str] = None,
        max_db_size: int = 2 * 1024 * 1024 * 1024,  # 2GB default
        enable_embeddings: bool = False,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.db_path = db_path or str(
            Path(__file__).parent.parent.parent / "data" / "kg_store"
        )
        self.max_db_size = max_db_size
        self.enable_embeddings = enable_embeddings
        self.embedding_model_name = embedding_model

        # In-memory graph
        self._init_networkx()

        # LMDB handle (lazy)
        self._lmdb_env = None

        # Embedding model (lazy)
        self._embedder = None

        # Stats
        self._stats = {
            "nodes_added": 0,
            "edges_added": 0,
            "queries": 0,
            "load_time": 0.0,
        }

        # Index: type → set of node_ids
        self._type_index: Dict[NodeType, Set[str]] = defaultdict(set)

        # Try to load from LMDB on init
        self._try_load()

    def _init_networkx(self):
        """Initialize NetworkX graph (lazy import to avoid circular deps)."""
        try:
            import networkx as nx
            self._graph = nx.DiGraph()
            self._nx = nx
        except ImportError:
            logger.warning("NetworkX not available — using dict-based fallback")
            self._graph = None
            self._nx = None
            self._adj: Dict[str, Dict[str, Dict]] = defaultdict(dict)
            self._nodes: Dict[str, Dict] = {}

    # ─── LMDB Backend ───────────────────────────────────────────────────────

    def _get_lmdb(self):
        """Lazy-init LMDB environment."""
        if self._lmdb_env is None:
            try:
                import lmdb
                os.makedirs(self.db_path, exist_ok=True)
                self._lmdb_env = lmdb.open(
                    self.db_path,
                    map_size=self.max_db_size,
                    max_dbs=4,
                )
            except ImportError:
                logger.warning("lmdb not installed — using JSON fallback")
                return None
            except Exception as e:
                logger.error(f"LMDB init failed: {e}")
                return None
        return self._lmdb_env

    def _try_load(self):
        """Load graph from LMDB if available."""
        start = time.time()
        env = self._get_lmdb()
        if env is None:
            # Try JSON fallback
            return self._try_load_json()

        try:
            with env.begin() as txn:
                # Load nodes
                nodes_data = txn.get(b"__nodes__")
                edges_data = txn.get(b"__edges__")

                if nodes_data:
                    nodes = pickle.loads(nodes_data)
                    for node_id, node_dict in nodes.items():
                        self._add_node_internal(node_id, node_dict)

                if edges_data:
                    edges = pickle.loads(edges_data)
                    for edge_key, edge_dict in edges.items():
                        src, tgt = edge_key.split("→", 1)
                        self._add_edge_internal(src, tgt, edge_dict)

            self._stats["load_time"] = time.time() - start
            if self.node_count > 0:
                logger.info(
                    f"KG loaded from LMDB: {self.node_count:,} nodes, "
                    f"{self.edge_count:,} edges in {self._stats['load_time']:.1f}s"
                )
        except Exception as e:
            logger.error(f"LMDB load failed: {e}")
            self._try_load_json()

    def _try_load_json(self):
        """Fallback: load from JSON file."""
        json_path = Path(self.db_path).parent / "kg_snapshot.json"
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text())
                for node in data.get("nodes", []):
                    self._add_node_internal(node["node_id"], node)
                for edge in data.get("edges", []):
                    self._add_edge_internal(edge["source_id"], edge["target_id"], edge)
                logger.info(f"KG loaded from JSON snapshot: {self.node_count} nodes")
            except Exception as e:
                logger.debug(f"JSON snapshot load failed: {e}")

    def save(self):
        """Persist current graph to LMDB (and JSON snapshot)."""
        env = self._get_lmdb()

        # Serialize all nodes and edges
        nodes_dict = {}
        edges_dict = {}

        if self._graph is not None:
            for node_id, data in self._graph.nodes(data=True):
                nodes_dict[node_id] = dict(data)
            for src, tgt, data in self._graph.edges(data=True):
                edges_dict[f"{src}→{tgt}"] = dict(data)
        else:
            nodes_dict = dict(self._nodes)
            for src, targets in self._adj.items():
                for tgt, data in targets.items():
                    edges_dict[f"{src}→{tgt}"] = data

        if env is not None:
            try:
                with env.begin(write=True) as txn:
                    txn.put(b"__nodes__", pickle.dumps(nodes_dict))
                    txn.put(b"__edges__", pickle.dumps(edges_dict))
                logger.info(f"KG saved to LMDB: {len(nodes_dict)} nodes, {len(edges_dict)} edges")
            except Exception as e:
                logger.error(f"LMDB save failed: {e}")

        # Always write JSON snapshot as fallback
        try:
            json_path = Path(self.db_path).parent / "kg_snapshot.json"
            snapshot = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "node_count": len(nodes_dict),
                "edge_count": len(edges_dict),
                "nodes": [{"node_id": k, **v} for k, v in list(nodes_dict.items())[:50000]],
                "edges": [
                    {"source_id": k.split("→")[0], "target_id": k.split("→")[1], **v}
                    for k, v in list(edges_dict.items())[:100000]
                ],
            }
            json_path.write_text(json.dumps(snapshot, indent=1, default=str))
            logger.info(f"KG JSON snapshot: {json_path}")
        except Exception as e:
            logger.debug(f"JSON snapshot write failed: {e}")

    # ─── Node Operations ────────────────────────────────────────────────────

    def add_node(
        self,
        node_id: str,
        node_type: NodeType | str = NodeType.TOOL,
        label: str = "",
        source: str = "",
        **properties,
    ) -> str:
        """Add a node to the graph. Returns the node_id."""
        if isinstance(node_type, str):
            try:
                node_type = NodeType(node_type.lower())
            except ValueError:
                node_type = NodeType.TOOL

        node_dict = {
            "node_type": node_type.value,
            "label": label or node_id,
            "source": source,
            **properties,
        }
        self._add_node_internal(node_id, node_dict)
        self._stats["nodes_added"] += 1
        return node_id

    def _add_node_internal(self, node_id: str, data: Dict):
        """Internal: add node to in-memory graph."""
        node_type_str = data.get("node_type", "tool")
        try:
            nt = NodeType(node_type_str)
        except ValueError:
            nt = NodeType.TOOL

        self._type_index[nt].add(node_id)

        if self._graph is not None:
            self._graph.add_node(node_id, **data)
        else:
            self._nodes[node_id] = data

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node data by ID."""
        self._stats["queries"] += 1
        if self._graph is not None:
            if node_id in self._graph:
                return dict(self._graph.nodes[node_id])
        else:
            return self._nodes.get(node_id)
        return None

    def has_node(self, node_id: str) -> bool:
        if self._graph is not None:
            return node_id in self._graph
        return node_id in self._nodes

    def get_nodes_by_type(self, node_type: NodeType | str) -> List[str]:
        """Get all node IDs of a given type."""
        if isinstance(node_type, str):
            try:
                node_type = NodeType(node_type.lower())
            except ValueError:
                return []
        return list(self._type_index.get(node_type, set()))

    # ─── Edge Operations ────────────────────────────────────────────────────

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType | str = EdgeType.LEADS_TO,
        weight: float = 1.0,
        source: str = "",
        **properties,
    ):
        """Add a directed edge between two nodes."""
        if isinstance(edge_type, str):
            try:
                edge_type = EdgeType(edge_type.lower())
            except ValueError:
                edge_type = EdgeType.LEADS_TO

        edge_dict = {
            "edge_type": edge_type.value,
            "weight": weight,
            "source": source,
            **properties,
        }
        self._add_edge_internal(source_id, target_id, edge_dict)
        self._stats["edges_added"] += 1

    def _add_edge_internal(self, src: str, tgt: str, data: Dict):
        """Internal: add edge to in-memory graph."""
        if self._graph is not None:
            self._graph.add_edge(src, tgt, **data)
        else:
            self._adj[src][tgt] = data

    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[EdgeType | str] = None,
        direction: str = "out",
    ) -> List[Tuple[str, Dict]]:
        """Get neighboring nodes, optionally filtered by edge type."""
        self._stats["queries"] += 1
        results = []

        if self._graph is not None:
            if direction == "out":
                edges = self._graph.out_edges(node_id, data=True)
            elif direction == "in":
                edges = self._graph.in_edges(node_id, data=True)
            else:
                edges = list(self._graph.out_edges(node_id, data=True)) + \
                        list(self._graph.in_edges(node_id, data=True))

            for edge_data in edges:
                if direction == "in":
                    src, _, data = edge_data
                    neighbor = src
                else:
                    _, tgt, data = edge_data
                    neighbor = tgt

                if edge_type:
                    et = edge_type.value if isinstance(edge_type, EdgeType) else edge_type
                    if data.get("edge_type") != et:
                        continue
                results.append((neighbor, data))
        else:
            if direction in ("out", "both"):
                for tgt, data in self._adj.get(node_id, {}).items():
                    if edge_type:
                        et = edge_type.value if isinstance(edge_type, EdgeType) else edge_type
                        if data.get("edge_type") != et:
                            continue
                    results.append((tgt, data))

        return results

    # ─── High-Level Queries ─────────────────────────────────────────────────

    def get_exploits_for_service(
        self, service_name: str, port: int = 0
    ) -> List[Dict[str, Any]]:
        """Find all exploits/techniques that target a service."""
        results = []
        # Build service node_id candidates
        candidates = []
        if port > 0:
            candidates.append(f"service:{service_name}:{port}")
            candidates.append(f"service:{port}")
        candidates.append(f"service:{service_name}")

        for svc_id in candidates:
            if not self.has_node(svc_id):
                continue
            # Get incoming FOUND_ON and EXPLOITS edges
            for neighbor, data in self.get_neighbors(svc_id, direction="in"):
                node_data = self.get_node(neighbor)
                if node_data:
                    results.append({
                        "node_id": neighbor,
                        "edge_type": data.get("edge_type", ""),
                        **node_data,
                    })

        return results

    def get_techniques_for_phase(self, phase: str) -> List[Dict[str, Any]]:
        """Get ATT&CK techniques relevant to a kill chain phase."""
        results = []
        phase_id = f"phase:{phase.upper()}"

        for neighbor, data in self.get_neighbors(phase_id, direction="in"):
            node_data = self.get_node(neighbor)
            if node_data and node_data.get("node_type") == "technique":
                results.append({"node_id": neighbor, **node_data})

        return results

    def shortest_attack_path(
        self, source_id: str, target_id: str, max_depth: int = 10
    ) -> Optional[List[str]]:
        """Find shortest attack path between two nodes."""
        if self._graph is None or self._nx is None:
            return None
        try:
            path = self._nx.shortest_path(
                self._graph, source_id, target_id,
            )
            return path if len(path) <= max_depth else None
        except (self._nx.NetworkXNoPath, self._nx.NodeNotFound):
            return None

    def get_attack_surface(self, host_id: str) -> Dict[str, Any]:
        """Get complete attack surface for a host — services, CVEs, techniques."""
        surface = {
            "host": host_id,
            "services": [],
            "cves": [],
            "techniques": [],
            "exploits": [],
            "credentials": [],
        }

        # Get services on this host
        for svc_id, edge_data in self.get_neighbors(host_id, EdgeType.HOSTS, "out"):
            svc_data = self.get_node(svc_id)
            if svc_data:
                surface["services"].append({"node_id": svc_id, **svc_data})

                # Get CVEs/exploits for each service
                for vuln_id, vuln_edge in self.get_neighbors(svc_id, direction="in"):
                    vuln_data = self.get_node(vuln_id)
                    if not vuln_data:
                        continue
                    vt = vuln_data.get("node_type", "")
                    if vt == "cve":
                        surface["cves"].append({"node_id": vuln_id, **vuln_data})
                    elif vt == "exploit":
                        surface["exploits"].append({"node_id": vuln_id, **vuln_data})
                    elif vt == "technique":
                        surface["techniques"].append({"node_id": vuln_id, **vuln_data})
                    elif vt == "credential":
                        surface["credentials"].append({"node_id": vuln_id, **vuln_data})

        return surface

    def query_subgraph(
        self,
        center_id: str,
        depth: int = 2,
        max_nodes: int = 100,
    ) -> Dict[str, Any]:
        """Extract a subgraph around a center node."""
        visited = set()
        queue = [(center_id, 0)]
        nodes = []
        edges = []

        while queue and len(visited) < max_nodes:
            current, d = queue.pop(0)
            if current in visited or d > depth:
                continue
            visited.add(current)

            node_data = self.get_node(current)
            if node_data:
                nodes.append({"node_id": current, **node_data})

            if d < depth:
                for neighbor, edge_data in self.get_neighbors(current, direction="both"):
                    edges.append({
                        "source": current,
                        "target": neighbor,
                        **edge_data,
                    })
                    if neighbor not in visited:
                        queue.append((neighbor, d + 1))

        return {"nodes": nodes, "edges": edges, "center": center_id, "depth": depth}

    # ─── Bulk Loading from Knowledge Base ────────────────────────────────────

    def load_from_knowledge_base(self, kb_dir: Optional[str] = None):
        """
        Bulk-load the knowledge graph from ingested JSON knowledge base files.

        Reads: services.json, cves.json, techniques.json, binaries.json,
               exploitdb.json, ad_attacks.json, commands.json
        """
        kb_path = Path(kb_dir) if kb_dir else (
            Path(__file__).parent.parent.parent / "data" / "knowledge_base"
        )

        if not kb_path.exists():
            logger.warning(f"Knowledge base not found at {kb_path}")
            return

        start = time.time()
        logger.info(f"Loading knowledge graph from {kb_path}...")

        # Create phase nodes
        phases = [
            "RECON", "ENUMERATION", "EXPLOITATION", "PRIVILEGE_ESCALATION",
            "LATERAL_MOVEMENT", "POST_EXPLOITATION", "EXFILTRATION",
        ]
        for i, phase in enumerate(phases):
            self.add_node(f"phase:{phase}", NodeType.PHASE, label=phase, order=i)
            if i > 0:
                self.add_edge(f"phase:{phases[i-1]}", f"phase:{phase}",
                              EdgeType.LEADS_TO, source="system")

        # Load services
        services_file = kb_path / "services.json"
        if services_file.exists():
            services = json.loads(services_file.read_text())
            if isinstance(services, list):
                for svc in services:
                    port = svc.get("port", 0)
                    name = svc.get("service_name", svc.get("name", ""))
                    svc_id = f"service:{name}:{port}" if port else f"service:{name}"
                    self.add_node(
                        svc_id, NodeType.SERVICE,
                        label=f"{name}:{port}" if port else name,
                        port=port,
                        service_name=name,
                        version=svc.get("version", ""),
                        description=svc.get("description", "")[:500],
                        source="knowledge_base",
                    )
            logger.info(f"  Services loaded: {len(self._type_index[NodeType.SERVICE])}")

        # Load CVEs
        cves_file = kb_path / "cves.json"
        if cves_file.exists():
            cves = json.loads(cves_file.read_text())
            if isinstance(cves, list):
                for cve in cves:
                    cve_id = cve.get("cve_id", "")
                    if not cve_id:
                        continue
                    self.add_node(
                        f"cve:{cve_id}", NodeType.CVE,
                        label=cve_id,
                        severity=cve.get("severity", ""),
                        description=cve.get("description", "")[:500],
                        affected_software=cve.get("affected_software", ""),
                        metasploit_module=cve.get("metasploit_module", ""),
                        source="knowledge_base",
                    )

                    # Link CVE to service if we can infer it
                    affected = cve.get("affected_software", "").lower()
                    for svc_id in self._type_index.get(NodeType.SERVICE, set()):
                        svc_data = self.get_node(svc_id)
                        if svc_data:
                            svc_name = svc_data.get("service_name", "").lower()
                            if svc_name and svc_name in affected:
                                self.add_edge(
                                    f"cve:{cve_id}", svc_id,
                                    EdgeType.FOUND_ON, source="inference",
                                )
            logger.info(f"  CVEs loaded: {len(self._type_index[NodeType.CVE])}")

        # Load techniques (MITRE ATT&CK)
        tech_file = kb_path / "techniques.json"
        if tech_file.exists():
            techniques = json.loads(tech_file.read_text())
            if isinstance(techniques, list):
                for tech in techniques:
                    tid = tech.get("technique_id", "")
                    if not tid:
                        continue
                    self.add_node(
                        f"technique:{tid}", NodeType.TECHNIQUE,
                        label=f"{tid}: {tech.get('technique_name', '')}",
                        technique_name=tech.get("technique_name", ""),
                        tactic=tech.get("tactic", ""),
                        description=tech.get("description", "")[:500],
                        platforms=tech.get("platforms", []),
                        source=tech.get("source", "knowledge_base"),
                    )

                    # Link technique to phase based on tactic
                    tactic = tech.get("tactic", "").lower()
                    phase_map = {
                        "reconnaissance": "RECON",
                        "resource-development": "RECON",
                        "initial-access": "EXPLOITATION",
                        "execution": "EXPLOITATION",
                        "persistence": "POST_EXPLOITATION",
                        "privilege-escalation": "PRIVILEGE_ESCALATION",
                        "defense-evasion": "POST_EXPLOITATION",
                        "credential-access": "EXPLOITATION",
                        "discovery": "ENUMERATION",
                        "lateral-movement": "LATERAL_MOVEMENT",
                        "collection": "EXFILTRATION",
                        "command-and-control": "POST_EXPLOITATION",
                        "exfiltration": "EXFILTRATION",
                        "impact": "EXFILTRATION",
                    }
                    for tac_name, phase in phase_map.items():
                        if tac_name in tactic:
                            self.add_edge(
                                f"technique:{tid}", f"phase:{phase}",
                                EdgeType.PHASE_OF, source="mitre_mapping",
                            )
            logger.info(f"  Techniques loaded: {len(self._type_index[NodeType.TECHNIQUE])}")

        # Load binaries (GTFOBins/LOLBAS)
        bin_file = kb_path / "binaries.json"
        if bin_file.exists():
            binaries = json.loads(bin_file.read_text())
            if isinstance(binaries, list):
                for b in binaries:
                    bname = b.get("binary_name", "")
                    if not bname:
                        continue
                    platform = b.get("platform", "linux")
                    self.add_node(
                        f"binary:{bname}", NodeType.BINARY,
                        label=bname,
                        platform=platform,
                        functions=b.get("functions", []),
                        description=b.get("description", "")[:300],
                        source=b.get("source", "knowledge_base"),
                    )

                    # Link to PRIVILEGE_ESCALATION phase
                    funcs = b.get("functions", [])
                    for func in funcs:
                        ftype = func.get("type", "").lower() if isinstance(func, dict) else ""
                        if ftype in ("suid", "sudo", "capabilities"):
                            self.add_edge(
                                f"binary:{bname}", "phase:PRIVILEGE_ESCALATION",
                                EdgeType.PHASE_OF, source="gtfobins_mapping",
                            )
                            break
            logger.info(f"  Binaries loaded: {len(self._type_index[NodeType.BINARY])}")

        # Load ExploitDB
        edb_file = kb_path / "exploitdb.json"
        if edb_file.exists():
            exploits = json.loads(edb_file.read_text())
            if isinstance(exploits, list):
                # Only load first 20K to keep graph manageable
                for exp in exploits[:20000]:
                    eid = exp.get("edb_id", "")
                    if not eid:
                        continue
                    self.add_node(
                        f"exploit:EDB-{eid}", NodeType.EXPLOIT,
                        label=f"EDB-{eid}: {exp.get('description', '')[:100]}",
                        platform=exp.get("platform", ""),
                        exploit_type=exp.get("exploit_type", ""),
                        port=exp.get("port", 0),
                        source="exploitdb",
                    )

                    # Link to CVE if available
                    cve = exp.get("cve", "")
                    if cve and self.has_node(f"cve:{cve}"):
                        self.add_edge(
                            f"exploit:EDB-{eid}", f"cve:{cve}",
                            EdgeType.EXPLOITS, source="exploitdb",
                        )

                    # Link to service via port
                    port = exp.get("port", 0)
                    if port > 0:
                        for svc_id in self._type_index.get(NodeType.SERVICE, set()):
                            svc_data = self.get_node(svc_id)
                            if svc_data and svc_data.get("port") == port:
                                self.add_edge(
                                    f"exploit:EDB-{eid}", svc_id,
                                    EdgeType.FOUND_ON, source="exploitdb",
                                )
            logger.info(f"  Exploits loaded: {len(self._type_index[NodeType.EXPLOIT])}")

        # Load tools from commands.json
        cmds_file = kb_path / "commands.json"
        if cmds_file.exists():
            commands = json.loads(cmds_file.read_text())
            if isinstance(commands, list):
                tool_set = set()
                for cmd in commands:
                    tool = cmd.get("tool_name", "")
                    if tool and tool not in tool_set and len(tool) > 1:
                        tool_set.add(tool)
                        if not self.has_node(f"tool:{tool}"):
                            self.add_node(
                                f"tool:{tool}", NodeType.TOOL,
                                label=tool,
                                source="commands",
                            )
                        # Link to phase
                        phase = cmd.get("phase", "").upper()
                        if phase in phases:
                            self.add_edge(
                                f"tool:{tool}", f"phase:{phase}",
                                EdgeType.PHASE_OF, source="commands",
                            )
            logger.info(f"  Tools loaded: {len(self._type_index[NodeType.TOOL])}")

        elapsed = time.time() - start
        logger.info(
            f"KG build complete: {self.node_count:,} nodes, {self.edge_count:,} edges "
            f"in {elapsed:.1f}s"
        )

        # Save to LMDB
        self.save()

    # ─── Dynamic: Add Runtime Discoveries ────────────────────────────────────

    def add_discovered_host(
        self,
        ip: str,
        hostname: str = "",
        os_info: str = "",
    ) -> str:
        """Add a dynamically discovered host during training."""
        host_id = f"host:{ip}"
        self.add_node(
            host_id, NodeType.HOST,
            label=hostname or ip,
            ip=ip,
            hostname=hostname,
            os=os_info,
            source="runtime_discovery",
        )
        return host_id

    def add_discovered_service(
        self,
        host_ip: str,
        port: int,
        service_name: str = "",
        version: str = "",
    ) -> str:
        """Add a dynamically discovered service during training."""
        svc_id = f"service:{service_name}:{port}" if service_name else f"service:{port}"

        # Ensure host exists
        host_id = f"host:{host_ip}"
        if not self.has_node(host_id):
            self.add_discovered_host(host_ip)

        self.add_node(
            svc_id, NodeType.SERVICE,
            label=f"{service_name}:{port}" if service_name else str(port),
            port=port,
            service_name=service_name,
            version=version,
            source="runtime_discovery",
        )
        self.add_edge(host_id, svc_id, EdgeType.HOSTS, source="runtime_discovery")

        return svc_id

    def add_discovered_credential(
        self,
        username: str,
        password: str = "",
        service_id: str = "",
    ) -> str:
        """Add a dynamically discovered credential."""
        cred_id = f"credential:{username}:{password[:4]}***"
        self.add_node(
            cred_id, NodeType.CREDENTIAL,
            label=f"{username}:***",
            username=username,
            source="runtime_discovery",
        )
        if service_id and self.has_node(service_id):
            self.add_edge(cred_id, service_id, EdgeType.CREDENTIALS_FOR,
                          source="runtime_discovery")
        return cred_id

    # ─── Embedding-Based Semantic Search ─────────────────────────────────────

    def _get_embedder(self):
        """Lazy-load sentence-transformer model."""
        if self._embedder is None and self.enable_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            except ImportError:
                logger.warning("sentence-transformers not installed")
                self.enable_embeddings = False
            except Exception as e:
                logger.error(f"Embedding model load failed: {e}")
                self.enable_embeddings = False
        return self._embedder

    def semantic_search(
        self,
        query: str,
        node_types: Optional[List[NodeType]] = None,
        top_k: int = 10,
    ) -> List[Tuple[str, float, Dict]]:
        """Search nodes by semantic similarity to query string."""
        embedder = self._get_embedder()
        if embedder is None:
            # Fallback to keyword search
            return self._keyword_search(query, node_types, top_k)

        try:
            import numpy as np
            query_vec = embedder.encode(query, convert_to_numpy=True)

            # Get candidate nodes
            candidates = []
            target_types = node_types or list(NodeType)
            for nt in target_types:
                for nid in self._type_index.get(nt, set()):
                    candidates.append(nid)

            if not candidates:
                return []

            # Build text representations for candidates
            texts = []
            for nid in candidates:
                data = self.get_node(nid)
                if data:
                    text = f"{data.get('label', '')} {data.get('description', '')} {data.get('technique_name', '')}"
                    texts.append(text)
                else:
                    texts.append(nid)

            # Encode and compute similarities
            cand_vecs = embedder.encode(texts, convert_to_numpy=True, batch_size=256,
                                        show_progress_bar=False)
            similarities = np.dot(cand_vecs, query_vec) / (
                np.linalg.norm(cand_vecs, axis=1) * np.linalg.norm(query_vec) + 1e-8
            )

            # Sort and return top-k
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = []
            for idx in top_indices:
                nid = candidates[idx]
                score = float(similarities[idx])
                data = self.get_node(nid) or {}
                results.append((nid, score, data))

            return results

        except Exception as e:
            logger.debug(f"Semantic search failed: {e}")
            return self._keyword_search(query, node_types, top_k)

    def _keyword_search(
        self,
        query: str,
        node_types: Optional[List[NodeType]] = None,
        top_k: int = 10,
    ) -> List[Tuple[str, float, Dict]]:
        """Fallback keyword-based search."""
        query_lower = query.lower()
        keywords = set(query_lower.split())
        results = []

        target_types = node_types or list(NodeType)
        for nt in target_types:
            for nid in self._type_index.get(nt, set()):
                data = self.get_node(nid)
                if not data:
                    continue
                text = f"{nid} {data.get('label', '')} {data.get('description', '')}".lower()
                score = sum(1.0 for kw in keywords if kw in text) / max(len(keywords), 1)
                if score > 0:
                    results.append((nid, score, data))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # ─── Properties ─────────────────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        if self._graph is not None:
            return self._graph.number_of_nodes()
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        if self._graph is not None:
            return self._graph.number_of_edges()
        return sum(len(targets) for targets in self._adj.values())

    @property
    def stats(self) -> Dict[str, Any]:
        type_counts = {nt.value: len(ids) for nt, ids in self._type_index.items() if ids}
        return {
            "nodes": self.node_count,
            "edges": self.edge_count,
            "types": type_counts,
            **self._stats,
        }

    def summary(self) -> str:
        """Human-readable summary of the graph."""
        s = self.stats
        lines = [
            f"Knowledge Graph: {s['nodes']:,} nodes, {s['edges']:,} edges",
        ]
        for nt, count in sorted(s.get("types", {}).items()):
            lines.append(f"  {nt}: {count:,}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<KnowledgeGraph nodes={self.node_count} edges={self.edge_count}>"
