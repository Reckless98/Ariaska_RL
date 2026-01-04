"""
Learned Commands - Store and retrieve commands that worked during attacks.

This module tracks successful command patterns from training runs,
allowing the system to learn which commands work in which contexts
and suggest them in future attacks.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from collections import defaultdict

from .command_registry import AttackPhase, CommandTemplate, COMMAND_REGISTRY


@dataclass
class LearnedCommand:
    """
    A command that was successfully used during an attack.
    
    Attributes:
        template_name: Name of the CommandTemplate used
        params: Parameters that were used
        context_tags: Tags describing the context (e.g., "linux", "htb_easy")
        success_count: How many times this worked
        total_attempts: Total times this was tried
        avg_reward: Average reward received
        discovered_at: When this was first learned
        last_used: When this was last successfully used
        notes: Any observations about when this works
        phase: Attack phase this belongs to
        preconditions_met: State flags that were true when this worked
    """
    template_name: str
    params: Dict[str, str]
    context_tags: Set[str] = field(default_factory=set)
    success_count: int = 0
    total_attempts: int = 0
    avg_reward: float = 0.0
    discovered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""
    phase: str = "RECON"
    preconditions_met: Set[str] = field(default_factory=set)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_attempts == 0:
            return 0.0
        return self.success_count / self.total_attempts
    
    @property
    def command_key(self) -> str:
        """Generate unique key for this command configuration."""
        # Sort params for consistent key
        params_str = "|".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.template_name}::{params_str}"
    
    def update_success(self, reward: float) -> None:
        """Update stats after successful use."""
        self.success_count += 1
        self.total_attempts += 1
        # Running average
        self.avg_reward = (
            (self.avg_reward * (self.success_count - 1) + reward) / self.success_count
        )
        self.last_used = datetime.now().isoformat()
    
    def update_failure(self) -> None:
        """Update stats after failed attempt."""
        self.total_attempts += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "template_name": self.template_name,
            "params": self.params,
            "context_tags": list(self.context_tags),
            "success_count": self.success_count,
            "total_attempts": self.total_attempts,
            "avg_reward": self.avg_reward,
            "discovered_at": self.discovered_at,
            "last_used": self.last_used,
            "notes": self.notes,
            "phase": self.phase,
            "preconditions_met": list(self.preconditions_met)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearnedCommand":
        """Create from dictionary."""
        return cls(
            template_name=data["template_name"],
            params=data.get("params", {}),
            context_tags=set(data.get("context_tags", [])),
            success_count=data.get("success_count", 0),
            total_attempts=data.get("total_attempts", 0),
            avg_reward=data.get("avg_reward", 0.0),
            discovered_at=data.get("discovered_at", datetime.now().isoformat()),
            last_used=data.get("last_used", datetime.now().isoformat()),
            notes=data.get("notes", ""),
            phase=data.get("phase", "RECON"),
            preconditions_met=set(data.get("preconditions_met", []))
        )


class LearnedCommandStore:
    """
    Persistent store for learned commands.
    
    Saves successful command patterns to disk and provides
    retrieval methods for suggesting commands in new attacks.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the learned command store.
        
        Args:
            storage_path: Path to JSON file for persistence.
                         Defaults to data/learned_commands.json
        """
        if storage_path is None:
            base_dir = Path(__file__).parent.parent.parent
            storage_path = base_dir / "data" / "learned_commands.json"
        
        self.storage_path = Path(storage_path)
        self.commands: Dict[str, LearnedCommand] = {}
        self._load()
    
    def _load(self) -> None:
        """Load commands from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    for key, cmd_data in data.get("commands", {}).items():
                        self.commands[key] = LearnedCommand.from_dict(cmd_data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load learned commands: {e}")
                self.commands = {}
    
    def _save(self) -> None:
        """Save commands to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "total_commands": len(self.commands),
            "commands": {
                key: cmd.to_dict() for key, cmd in self.commands.items()
            }
        }
        
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def record_success(
        self,
        template_name: str,
        params: Dict[str, str],
        reward: float,
        context_tags: Optional[Set[str]] = None,
        preconditions_met: Optional[Set[str]] = None,
        phase: Optional[AttackPhase] = None,
        notes: str = ""
    ) -> LearnedCommand:
        """
        Record a successful command use.
        
        Args:
            template_name: Name of the command template
            params: Parameters used
            reward: Reward received
            context_tags: Context tags (e.g., target type)
            preconditions_met: State flags that were true
            phase: Attack phase
            notes: Optional notes
            
        Returns:
            The updated LearnedCommand
        """
        # Create temporary command to get key
        temp_cmd = LearnedCommand(
            template_name=template_name,
            params=params
        )
        key = temp_cmd.command_key
        
        if key in self.commands:
            # Update existing
            cmd = self.commands[key]
            cmd.update_success(reward)
            if context_tags:
                cmd.context_tags.update(context_tags)
            if preconditions_met:
                cmd.preconditions_met.update(preconditions_met)
        else:
            # Create new
            cmd = LearnedCommand(
                template_name=template_name,
                params=params,
                context_tags=context_tags or set(),
                preconditions_met=preconditions_met or set(),
                phase=phase.name if phase else "RECON",
                notes=notes
            )
            cmd.update_success(reward)
            self.commands[key] = cmd
        
        self._save()
        return cmd
    
    def record_failure(
        self,
        template_name: str,
        params: Dict[str, str]
    ) -> Optional[LearnedCommand]:
        """
        Record a failed command attempt.
        
        Args:
            template_name: Name of the command template
            params: Parameters used
            
        Returns:
            The updated LearnedCommand if it exists, None otherwise
        """
        temp_cmd = LearnedCommand(template_name=template_name, params=params)
        key = temp_cmd.command_key
        
        if key in self.commands:
            self.commands[key].update_failure()
            self._save()
            return self.commands[key]
        
        return None
    
    def get_successful_commands(
        self,
        phase: Optional[AttackPhase] = None,
        context_tags: Optional[Set[str]] = None,
        min_success_rate: float = 0.5,
        min_successes: int = 1,
        limit: int = 10
    ) -> List[LearnedCommand]:
        """
        Get successful commands matching criteria.
        
        Args:
            phase: Filter by attack phase
            context_tags: Filter by context tags (any match)
            min_success_rate: Minimum success rate
            min_successes: Minimum number of successes
            limit: Maximum number to return
            
        Returns:
            List of matching LearnedCommands sorted by avg_reward
        """
        results = []
        
        for cmd in self.commands.values():
            # Phase filter
            if phase and cmd.phase != phase.name:
                continue
            
            # Context tag filter (any match)
            if context_tags and not cmd.context_tags.intersection(context_tags):
                continue
            
            # Success rate filter
            if cmd.success_rate < min_success_rate:
                continue
            
            # Minimum successes filter
            if cmd.success_count < min_successes:
                continue
            
            results.append(cmd)
        
        # Sort by average reward descending
        results.sort(key=lambda c: c.avg_reward, reverse=True)
        
        return results[:limit]
    
    def get_commands_for_prompt(
        self,
        phase: Optional[AttackPhase] = None,
        preconditions: Optional[Set[str]] = None,
        limit: int = 5
    ) -> List[str]:
        """
        Get learned command suggestions for LLM prompt.
        
        Args:
            phase: Current attack phase
            preconditions: Current state flags
            limit: Maximum suggestions
            
        Returns:
            List of command descriptions for prompt
        """
        # Get candidates
        candidates = self.get_successful_commands(
            phase=phase,
            min_success_rate=0.3,
            min_successes=1,
            limit=limit * 2  # Get extra, we'll filter
        )
        
        # Filter by preconditions if provided
        if preconditions:
            candidates = [
                c for c in candidates
                if c.preconditions_met.issubset(preconditions) or not c.preconditions_met
            ]
        
        # Format for prompt
        suggestions = []
        for cmd in candidates[:limit]:
            params_str = ", ".join(f"{k}={v}" for k, v in cmd.params.items())
            suggestions.append(
                f"✓ {cmd.template_name}({params_str}) - "
                f"worked {cmd.success_count}x (avg reward: {cmd.avg_reward:.1f})"
            )
        
        return suggestions
    
    def get_phase_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics per attack phase."""
        stats = defaultdict(lambda: {
            "total_commands": 0,
            "total_successes": 0,
            "avg_success_rate": 0.0,
            "top_commands": []
        })
        
        for cmd in self.commands.values():
            phase = cmd.phase
            stats[phase]["total_commands"] += 1
            stats[phase]["total_successes"] += cmd.success_count
        
        # Calculate averages and top commands
        for phase, phase_stats in stats.items():
            phase_cmds = [c for c in self.commands.values() if c.phase == phase]
            if phase_cmds:
                phase_stats["avg_success_rate"] = sum(
                    c.success_rate for c in phase_cmds
                ) / len(phase_cmds)
                
                # Top 3 by reward
                top = sorted(phase_cmds, key=lambda c: c.avg_reward, reverse=True)[:3]
                phase_stats["top_commands"] = [c.template_name for c in top]
        
        return dict(stats)
    
    def promote_to_registry(
        self,
        command_key: str,
        min_successes: int = 5,
        min_success_rate: float = 0.7
    ) -> Optional[CommandTemplate]:
        """
        Promote a learned command to the main registry if it's proven.
        
        This is for commands that agents discover that aren't in the
        original registry but should be added.
        
        Args:
            command_key: Key of the learned command
            min_successes: Minimum successes required
            min_success_rate: Minimum success rate required
            
        Returns:
            New CommandTemplate if promoted, None otherwise
        """
        if command_key not in self.commands:
            return None
        
        cmd = self.commands[command_key]
        
        # Check if meets promotion criteria
        if cmd.success_count < min_successes:
            return None
        if cmd.success_rate < min_success_rate:
            return None
        
        # Check if already in registry
        if cmd.template_name in COMMAND_REGISTRY:
            return None
        
        # Create template from learned command
        # This is for truly novel commands discovered by agents
        phase = AttackPhase[cmd.phase] if cmd.phase in AttackPhase.__members__ else AttackPhase.RECON
        
        # We can't easily create a new template since we don't have
        # the full command string - this would need manual review
        # Just log it for now
        print(f"[LearnedCommands] Command ready for promotion: {cmd.template_name}")
        print(f"  Success rate: {cmd.success_rate:.2%} ({cmd.success_count}/{cmd.total_attempts})")
        print(f"  Avg reward: {cmd.avg_reward:.2f}")
        
        return None
    
    def clear_old_commands(self, days: int = 30) -> int:
        """
        Remove commands not used in the last N days.
        
        Args:
            days: Days of inactivity before removal
            
        Returns:
            Number of commands removed
        """
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.isoformat()
        
        to_remove = [
            key for key, cmd in self.commands.items()
            if cmd.last_used < cutoff_str
        ]
        
        for key in to_remove:
            del self.commands[key]
        
        if to_remove:
            self._save()
        
        return len(to_remove)
    
    def export_successful_chains(
        self,
        min_chain_length: int = 3
    ) -> List[List[LearnedCommand]]:
        """
        Export sequences of successful commands (attack chains).
        
        This helps identify patterns like:
        nmap -> gobuster -> sqli -> shell
        
        Returns:
            List of command chains (lists of LearnedCommands)
        """
        # Group by session/time proximity
        # For now, just sort by discovery time and group by phase order
        successful = [c for c in self.commands.values() if c.success_count > 0]
        
        if len(successful) < min_chain_length:
            return []
        
        # Sort by discovery time
        successful.sort(key=lambda c: c.discovered_at)
        
        # Build chains (simplified - just phase order)
        chains = []
        current_chain = []
        
        phase_order = [p.name for p in AttackPhase]
        
        for cmd in successful:
            if not current_chain:
                current_chain.append(cmd)
            else:
                # Check if this follows the attack chain
                last_phase_idx = phase_order.index(current_chain[-1].phase)
                curr_phase_idx = phase_order.index(cmd.phase)
                
                if curr_phase_idx >= last_phase_idx:
                    current_chain.append(cmd)
                else:
                    # Start new chain
                    if len(current_chain) >= min_chain_length:
                        chains.append(current_chain)
                    current_chain = [cmd]
        
        # Don't forget the last chain
        if len(current_chain) >= min_chain_length:
            chains.append(current_chain)
        
        return chains
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.commands:
            return {
                "total_learned": 0,
                "total_successes": 0,
                "by_phase": {}
            }
        
        return {
            "total_learned": len(self.commands),
            "total_successes": sum(c.success_count for c in self.commands.values()),
            "total_attempts": sum(c.total_attempts for c in self.commands.values()),
            "overall_success_rate": (
                sum(c.success_count for c in self.commands.values()) /
                max(sum(c.total_attempts for c in self.commands.values()), 1)
            ),
            "avg_reward": (
                sum(c.avg_reward for c in self.commands.values()) / len(self.commands)
            ),
            "by_phase": self.get_phase_statistics()
        }


# Global store instance
_store: Optional[LearnedCommandStore] = None


def get_learned_store(storage_path: Optional[str] = None) -> LearnedCommandStore:
    """Get or create the global learned command store."""
    global _store
    if _store is None:
        _store = LearnedCommandStore(storage_path)
    return _store


def reset_learned_store() -> None:
    """Reset the global store (mainly for testing)."""
    global _store
    _store = None
