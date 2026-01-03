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
        Find skills matching condition keywords.
        
        Args:
            condition_keywords: Keywords to match in if_condition
            
        Returns:
            List of matching skills
        """
        matches = []
        for skill in self.skills.values():
            condition_lower = skill.if_condition.lower()
            if any(kw.lower() in condition_lower for kw in condition_keywords):
                matches.append(skill)
        
        # Sort by confidence
        matches.sort(key=lambda s: s.confidence, reverse=True)
        return matches
    
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
