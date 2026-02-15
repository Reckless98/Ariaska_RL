"""
KnowledgeCandidate v2 — Full Envelope Object for Evidence-Gated Attack Knowledge.

Phase 10.0 — Knowledge Governance.

Every KnowledgeCandidate v2 object contains ALL information needed for:
- Selection (taxonomy, phase_fit, tags)
- Evidence gating (prerequisites, anti-requirements, confidence)
- Safe execution via registry templates ONLY
- Verification (success/failure indicators)
- Governance/telemetry (allowed scopes, quality metrics)
- Future reasoning (raw preservation, references)

ABSOLUTE RULE: No candidate may carry raw shell as executable output.
Execution must map to COMMAND_REGISTRY templates only.
Raw shell exists only in raw_preservation.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.knowledge.candidate_v2")

SCHEMA_VERSION = "2.0.0"


# ─── Sub-schemas ──────────────────────────────────────────────────────────────

@dataclass
class SourceInfo:
    """Provenance tracking for a knowledge candidate."""
    origin: str = ""            # wadcoms|htb|tryhackme|cve|exploitdb|hacktricks|gtfobins|lolbas|patt|nuclei|metasploit|custom|atomic-red-team
    path: str = ""              # file path or URL
    raw_ref: str = ""           # original key/name in source
    ingested_at: str = ""       # ISO timestamp
    source_hash: str = ""       # SHA256 of source entry for change detection


@dataclass
class MitreMapping:
    """MITRE ATT&CK technique mapping."""
    technique: str = ""         # Txxxx
    tactic: str = ""            # TAxxxx


@dataclass
class PlatformInfo:
    """Target platform requirements."""
    os: List[str] = field(default_factory=lambda: ["linux"])
    arch: List[str] = field(default_factory=lambda: ["x86", "x64"])
    requires_gui: bool = False


@dataclass
class Taxonomy:
    """Classification and categorization."""
    vuln_family: str = ""                   # e.g., "authentication-bypass", "rce", "info-disclosure"
    service_archetype: str = ""             # e.g., "smb", "web-app", "ssh", "database"
    exploit_archetype: str = ""             # e.g., "default-creds", "command-injection", "buffer-overflow"
    killchain_step: str = ""                # recon|foothold|privesc|lateral|persist|exfil
    phase_fit: List[str] = field(default_factory=list)  # internal AttackPhase names
    mitre: MitreMapping = field(default_factory=MitreMapping)
    platform: PlatformInfo = field(default_factory=PlatformInfo)
    tags: List[str] = field(default_factory=list)


@dataclass
class EvidenceGate:
    """Evidence requirements for candidate selection."""
    evidence_requirements: List[str] = field(default_factory=list)   # what must be observed
    prerequisites: List[str] = field(default_factory=list)            # flags that must be set
    anti_requirements: List[str] = field(default_factory=list)        # conditions that block use
    confidence: float = 0.5
    risk: float = 0.5
    detection_risk: float = 0.3


@dataclass
class ExecutionStep:
    """A single step in a multi-step execution sequence."""
    template: str = ""          # registry template name
    why: str = ""               # rationale
    expects: str = ""           # expected outcome


@dataclass
class Verification:
    """Post-execution verification."""
    template: str = ""          # registry template name for verification
    expects: str = ""           # expected verification result


@dataclass
class Execution:
    """Safe execution plan via registry templates ONLY."""
    command_templates: List[str] = field(default_factory=list)        # registry template names
    parameters: Dict[str, str] = field(default_factory=dict)
    sequencing: List[ExecutionStep] = field(default_factory=list)
    not_when: List[str] = field(default_factory=list)
    success_indicators: List[str] = field(default_factory=list)
    failure_indicators: List[str] = field(default_factory=list)
    verification: Optional[Verification] = None


@dataclass
class References:
    """External references and documentation."""
    cves: List[str] = field(default_factory=list)
    advisories: List[str] = field(default_factory=list)
    papers: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    modules: List[str] = field(default_factory=list)


@dataclass
class QualityMetrics:
    """Quality assessment of the candidate."""
    template_mappable: bool = False
    evidence_coverage: float = 0.0      # 0.0-1.0
    prereq_coverage: float = 0.0        # 0.0-1.0


@dataclass
class Governance:
    """Safety and governance constraints."""
    allowed_scopes: List[str] = field(default_factory=lambda: ["htb", "tryhackme", "lab", "ctf"])
    safety_constraints: List[str] = field(default_factory=lambda: ["registry_only", "evidence_gated", "no_freeform"])
    quality: QualityMetrics = field(default_factory=QualityMetrics)


@dataclass
class RuntimeStats:
    """Runtime usage statistics (updated during training)."""
    times_offered: int = 0
    times_selected: int = 0
    times_succeeded: int = 0


@dataclass
class RawPreservation:
    """Original source material preserved for audit."""
    original_text: str = ""
    original_commands: List[str] = field(default_factory=list)
    notes: str = ""


# ─── Main Schema ──────────────────────────────────────────────────────────────

@dataclass
class KnowledgeCandidate:
    """
    KnowledgeCandidate v2 — Full Envelope Object.

    Contains ALL information needed for selection, evidence gating,
    safe execution via registry templates, verification, governance,
    and future reasoning.
    """
    candidate_id: str = ""
    schema_version: str = SCHEMA_VERSION
    title: str = ""
    summary: str = ""

    source: SourceInfo = field(default_factory=SourceInfo)
    taxonomy: Taxonomy = field(default_factory=Taxonomy)
    evidence_gate: EvidenceGate = field(default_factory=EvidenceGate)
    execution: Execution = field(default_factory=Execution)
    references: References = field(default_factory=References)
    governance: Governance = field(default_factory=Governance)
    runtime_stats: RuntimeStats = field(default_factory=RuntimeStats)
    raw_preservation: RawPreservation = field(default_factory=RawPreservation)

    def generate_id(self) -> str:
        """Generate stable candidate_id from source + title."""
        key = f"{self.source.origin}:{self.source.raw_ref}:{self.title}"
        self.candidate_id = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.candidate_id

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON output."""
        d = asdict(self)
        # Convert nested dataclasses that might be None
        if d.get("execution", {}).get("verification") is None:
            d["execution"]["verification"] = None
        return d

    def to_json_line(self) -> str:
        """Serialize to single JSON line for JSONL output."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeCandidate":
        """Deserialize from dict."""
        kc = cls()
        kc.candidate_id = data.get("candidate_id", "")
        kc.schema_version = data.get("schema_version", SCHEMA_VERSION)
        kc.title = data.get("title", "")
        kc.summary = data.get("summary", "")

        src = data.get("source", {})
        kc.source = SourceInfo(**{k: src.get(k, "") for k in SourceInfo.__dataclass_fields__})

        tax = data.get("taxonomy", {})
        mitre_d = tax.get("mitre", {})
        platform_d = tax.get("platform", {})
        kc.taxonomy = Taxonomy(
            vuln_family=tax.get("vuln_family", ""),
            service_archetype=tax.get("service_archetype", ""),
            exploit_archetype=tax.get("exploit_archetype", ""),
            killchain_step=tax.get("killchain_step", ""),
            phase_fit=tax.get("phase_fit", []),
            mitre=MitreMapping(**{k: mitre_d.get(k, "") for k in MitreMapping.__dataclass_fields__}),
            platform=PlatformInfo(
                os=platform_d.get("os", ["linux"]),
                arch=platform_d.get("arch", ["x86", "x64"]),
                requires_gui=platform_d.get("requires_gui", False),
            ),
            tags=tax.get("tags", []),
        )

        eg = data.get("evidence_gate", {})
        kc.evidence_gate = EvidenceGate(
            evidence_requirements=eg.get("evidence_requirements", []),
            prerequisites=eg.get("prerequisites", []),
            anti_requirements=eg.get("anti_requirements", []),
            confidence=eg.get("confidence", 0.5),
            risk=eg.get("risk", 0.5),
            detection_risk=eg.get("detection_risk", 0.3),
        )

        ex = data.get("execution", {})
        seq_list = []
        for s in ex.get("sequencing", []):
            seq_list.append(ExecutionStep(
                template=s.get("template", ""),
                why=s.get("why", ""),
                expects=s.get("expects", ""),
            ))
        verif = ex.get("verification")
        verif_obj = None
        if verif and isinstance(verif, dict):
            verif_obj = Verification(
                template=verif.get("template", ""),
                expects=verif.get("expects", ""),
            )
        kc.execution = Execution(
            command_templates=ex.get("command_templates", []),
            parameters=ex.get("parameters", {}),
            sequencing=seq_list,
            not_when=ex.get("not_when", []),
            success_indicators=ex.get("success_indicators", []),
            failure_indicators=ex.get("failure_indicators", []),
            verification=verif_obj,
        )

        ref = data.get("references", {})
        kc.references = References(
            cves=ref.get("cves", []),
            advisories=ref.get("advisories", []),
            papers=ref.get("papers", []),
            tools=ref.get("tools", []),
            modules=ref.get("modules", []),
        )

        gov = data.get("governance", {})
        qual = gov.get("quality", {})
        kc.governance = Governance(
            allowed_scopes=gov.get("allowed_scopes", ["htb", "tryhackme", "lab", "ctf"]),
            safety_constraints=gov.get("safety_constraints", ["registry_only", "evidence_gated", "no_freeform"]),
            quality=QualityMetrics(
                template_mappable=qual.get("template_mappable", False),
                evidence_coverage=qual.get("evidence_coverage", 0.0),
                prereq_coverage=qual.get("prereq_coverage", 0.0),
            ),
        )

        rs = data.get("runtime_stats", {})
        kc.runtime_stats = RuntimeStats(
            times_offered=rs.get("times_offered", 0),
            times_selected=rs.get("times_selected", 0),
            times_succeeded=rs.get("times_succeeded", 0),
        )

        rp = data.get("raw_preservation", {})
        kc.raw_preservation = RawPreservation(
            original_text=rp.get("original_text", ""),
            original_commands=rp.get("original_commands", []),
            notes=rp.get("notes", ""),
        )

        return kc


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_candidates_jsonl(path: str) -> List[KnowledgeCandidate]:
    """Load KnowledgeCandidate v2 objects from a JSONL file."""
    candidates = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    candidates.append(KnowledgeCandidate.from_dict(data))
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning(f"Failed to parse line {line_num} in {path}: {e}")
    except FileNotFoundError:
        logger.warning(f"JSONL file not found: {path}")
    return candidates


def save_candidates_jsonl(candidates: List[KnowledgeCandidate], path: str) -> int:
    """Save KnowledgeCandidate v2 objects to a JSONL file. Returns count written."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(c.to_json_line() + "\n")
            count += 1
    logger.info(f"Wrote {count} candidates to {path}")
    return count
