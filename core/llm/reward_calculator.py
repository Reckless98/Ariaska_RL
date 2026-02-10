"""
Smart Reward Calculator - Intelligent reward shaping for better learning.

This module provides sophisticated reward calculation that encourages
exploration, phase progression, and penalizes redundant actions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict
from datetime import datetime
import math

from ..commands.command_registry import (
    AttackPhase,
    CommandTemplate,
    COMMAND_REGISTRY,
    get_phase_from_state
)


@dataclass
class RewardBreakdown:
    """
    Detailed breakdown of how reward was calculated.
    
    Attributes:
        base_reward: Base reward from the action
        novelty_bonus: Bonus for trying something new
        progress_bonus: Bonus for making progress
        phase_advance_bonus: Bonus for advancing to new phase
        redundancy_penalty: Penalty for repeating commands
        failure_penalty: Penalty for failed commands
        discovery_bonus: Bonus for finding new information
        efficiency_bonus: Bonus for efficient command use
        total: Final calculated reward
        explanation: Human-readable explanation
    """
    base_reward: float = 0.0
    novelty_bonus: float = 0.0
    progress_bonus: float = 0.0
    phase_advance_bonus: float = 0.0
    redundancy_penalty: float = 0.0
    failure_penalty: float = 0.0
    discovery_bonus: float = 0.0
    efficiency_bonus: float = 0.0
    total: float = 0.0
    explanation: str = ""
    
    def calculate_total(self) -> float:
        """Calculate the total reward from components with a per-step floor.

        Industry standard: a floor prevents catastrophic negative spirals
        that make PPO's value function diverge.  The agent still feels
        pain from redundancy, but not unbounded pain.
        """
        raw_total = (
            self.base_reward +
            self.novelty_bonus +
            self.progress_bonus +
            self.phase_advance_bonus +
            self.discovery_bonus +
            self.efficiency_bonus -
            self.redundancy_penalty -
            self.failure_penalty
        )
        # PHASE 6: Floor at -5.0, ceiling at 50.0 per step
        # Tighter ceiling (was 100.0) prevents extreme outlier rewards
        # from destabilizing PPO value function. Max single-step now ~50
        # instead of ~655, bringing the reward range closer to value
        # function capacity.
        self.total = max(min(raw_total, 50.0), -5.0)
        return self.total
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "base_reward": self.base_reward,
            "novelty_bonus": self.novelty_bonus,
            "progress_bonus": self.progress_bonus,
            "phase_advance_bonus": self.phase_advance_bonus,
            "redundancy_penalty": self.redundancy_penalty,
            "failure_penalty": self.failure_penalty,
            "discovery_bonus": self.discovery_bonus,
            "efficiency_bonus": self.efficiency_bonus,
            "total": self.total
        }


class SmartRewardCalculator:
    """
    Intelligent reward calculator with shaping for better RL learning.
    
    Tracks command history, discoveries, and phase progression to
    provide informative reward signals.
    """
    
    # Phase progression rewards (cumulative as you advance)
    PHASE_REWARDS = {
        AttackPhase.RECON: 0.0,
        AttackPhase.ENUMERATION: 5.0,
        AttackPhase.EXPLOITATION: 15.0,
        AttackPhase.PRIVILEGE_ESCALATION: 30.0,
        AttackPhase.LATERAL_MOVEMENT: 45.0,
        AttackPhase.POST_EXPLOITATION: 60.0,
        AttackPhase.EXFILTRATION: 75.0,
        AttackPhase.CLOSEOUT: 90.0,
    }
    
    # Discovery type bonuses — Phase 5.1: +30% boost for genuine achievements
    # These reward ACTUAL discoveries, not just existing
    DISCOVERY_BONUSES = {
        # Port/Service discoveries
        "open_port": 2.5,       # ports are valuable in recon
        "service": 5.0,         # service identification is key
        "version": 6.5,         # version = exploit potential
        
        # User/Credential discoveries
        "user": 8.0,            # user discovery
        "username": 8.0,        # users are high value
        "password": 26.0,       # passwords are gold
        "hash": 16.0,           # hashes can be cracked
        "credential": 20.0,     # any credential is valuable
        
        # Vulnerability discoveries
        "vulnerability": 10.0,  # vulns lead to exploitation
        "cve": 13.0,            # specific CVE = exploit ready
        
        # Shell/Access discoveries — Phase 6: reduced to fit 50.0 ceiling
        "shell": 25.0,          # shell = major milestone (was 50)
        "root_shell": 45.0,     # root = game over (was 130)
        "flag": 50.0,           # CTF flag = ultimate goal (was 200)
        
        # Web discoveries
        "directory": 2.5,       # directories expand attack surface
        "web_path": 4.0,        # specific interesting paths
        "file": 2.5,            # files can be valuable
        "sensitive_file": 10.0, # sensitive files are high value
        
        # Network discoveries
        "subdomain": 5.0,       # subdomains expand attack surface
        "share": 6.5,           # shares can leak data
        "smb_share": 6.5,       # SMB shares specifically
        "domain_info": 6.5,     # domain info helps lateral movement
        
        # Database discoveries
        "database": 8.0,        # database identified
        "db_name": 5.0,         # specific database names
        
        # Phase 5: Additional discovery types
        "dns_record": 4.0,      # DNS records expand recon knowledge
        "web_parameter": 5.0,   # Injectable parameters are high value
        "api_endpoint": 6.5,    # API endpoints reveal attack surface
        "version_info": 4.5,    # Version info = exploit matching
    }
    
    def __init__(
        self,
        novelty_weight: float = 1.5,          # was 1.0 - encourages exploration
        redundancy_decay: float = 0.5,
        max_redundancy_penalty: float = 0.5,   # Very low - don't punish too harshly
        phase_advance_multiplier: float = 4.0, # Phase 5.1: 3→4, reward genuine progression
        efficiency_window: int = 10,
        progress_bonus_per_step: float = 1.0,  # Phase 5.1: Honest scaling (was 12.0)
        ms2_mode: bool = False,                # Phase 6.4: Enable MS2-specific reward shaping
        target_profile: str = "",              # Phase 6.9.5: "metasploitable2", "metasploitable3", etc.
    ):
        """
        Initialize the reward calculator.
        
        Args:
            novelty_weight: Weight for novelty bonus
            redundancy_decay: How much to decay redundancy penalty over time
            max_redundancy_penalty: Maximum penalty for redundant commands
            phase_advance_multiplier: Multiplier for phase advancement bonus
            efficiency_window: Window for calculating efficiency
            progress_bonus_per_step: Base bonus per step (1.0 = industry standard)
            ms2_mode: Enable Metasploitable 2 specific reward shaping
            target_profile: Target identifier for loading correct exploit graph
        """
        self.novelty_weight = novelty_weight
        self.redundancy_decay = redundancy_decay
        self.max_redundancy_penalty = max_redundancy_penalty
        self.phase_advance_multiplier = phase_advance_multiplier
        self.efficiency_window = efficiency_window
        self.progress_bonus_per_step = progress_bonus_per_step
        self.ms2_mode = ms2_mode
        self.target_profile = target_profile
        
        # Phase 6.9.5: Load exploit graph based on target_profile (preferred) or ms2_mode (legacy)
        self._exploit_graph = None
        if target_profile == "metasploitable3":
            try:
                from core.knowledge.ms3_exploit_graph import get_ms3_graph
                self._exploit_graph = get_ms3_graph()
            except ImportError:
                pass
        elif target_profile == "metasploitable2" or ms2_mode:
            try:
                from core.knowledge.ms2_exploit_graph import get_ms2_graph
                self._exploit_graph = get_ms2_graph()
            except ImportError:
                pass
        self._ms2_graph = self._exploit_graph  # Backwards compat alias
        
        # Tracking state
        self.command_history: List[str] = []
        self.template_usage: Dict[str, int] = defaultdict(int)
        self.discoveries: Set[str] = set()
        self.highest_phase: AttackPhase = AttackPhase.RECON
        self.phase_history: List[AttackPhase] = []
        self.reward_history: List[float] = []
        self.last_useful_command_idx: int = 0
        self.session_start: datetime = datetime.now()
    
    def reset(self) -> None:
        """Reset calculator state for new session."""
        self.command_history = []
        self.template_usage = defaultdict(int)
        self.discoveries = set()
        self.highest_phase = AttackPhase.RECON
        self.phase_history = []
        self.reward_history = []
        self.last_useful_command_idx = 0
        self.session_start = datetime.now()
    
    def calculate_reward(
        self,
        template_name: str,
        command: str,
        success: bool,
        raw_output: str,
        current_phase: AttackPhase,
        state_flags: Dict[str, bool],
        new_discoveries: Optional[Dict[str, Any]] = None,
        shared_discoveries: Optional[Set[str]] = None,
    ) -> RewardBreakdown:
        """
        Calculate reward for a command execution.
        
        Args:
            template_name: Name of command template used
            command: Actual command executed
            success: Whether command succeeded
            raw_output: Output from command
            current_phase: Current attack phase
            state_flags: Current state flags
            new_discoveries: New things discovered
            shared_discoveries: Shared set across all agents for dedup.
                If provided, novelty is tracked globally (not per-agent).
            
        Returns:
            RewardBreakdown with detailed reward components
        """
        breakdown = RewardBreakdown()
        explanations = []
        
        # Use shared discoveries if provided (Phase 6: cross-agent dedup)
        disc_tracker = shared_discoveries if shared_discoveries is not None else self.discoveries
        
        # Get template info
        template = COMMAND_REGISTRY.get(template_name)
        
        # REWARD MULTIPLIER: Scale POSITIVE rewards, keep penalties real
        # Phase 6: Reduced 2.5 → 1.5 to prevent reward inflation
        # Combined with tighter ceiling (50.0), keeps value function stable
        REWARD_MULTIPLIER = 1.5
        
        # 1. Base reward from template 
        if template:
            base = template.typical_reward if success else 0.0
            breakdown.base_reward = base * REWARD_MULTIPLIER
            explanations.append(f"Base: {breakdown.base_reward:.1f}")
        
        # 1b. Progress bonus - small reward for taking action (not guaranteed success)
        breakdown.progress_bonus = self.progress_bonus_per_step  # Phase 5.1: honest 1.0/step
        explanations.append(f"Progress: +{breakdown.progress_bonus:.1f}")
        
        # 2. Novelty bonus - reward trying NEW commands (calibrated for PPO)
        # Phase 4: Reduced from 8→5 to prevent novelty-seeking over objective progress
        if template_name not in self.template_usage:
            # First time using this command
            breakdown.novelty_bonus = 5.0 * self.novelty_weight
            explanations.append(f"🆕 Novelty (first use): +{breakdown.novelty_bonus:.1f}")
        elif self.template_usage[template_name] < 3:
            # Second or third use - smaller bonus
            bonus = (3 - self.template_usage[template_name]) * 1.5 * self.novelty_weight
            breakdown.novelty_bonus = bonus
            explanations.append(f"Novelty (rare): +{bonus:.1f}")
        
        # 3. Escalating redundancy penalty - Phase 5.1: scales linearly, no hard cap
        # 1st repeat=-3, 2nd=-6, 3rd=-9, ... up to soft cap -30
        # With progress_bonus=1.0, net is negative from FIRST repeat
        repeat_count = self.command_history.count(command)
        if repeat_count > 0:
            penalty = min(3.0 * repeat_count, 30.0)
            breakdown.redundancy_penalty = penalty
            if repeat_count >= 3:
                explanations.append(f"🔁 STUCK LOOP: -{penalty:.1f} (repeat #{repeat_count})")
            else:
                explanations.append(f"Redundancy: -{penalty:.1f}")
        
        # 4. Discovery bonuses - these are the REAL rewards
        if new_discoveries:
            total_discovery_bonus = 0.0
            for discovery_type, values in new_discoveries.items():
                if discovery_type in self.DISCOVERY_BONUSES:
                    # Count new discoveries — use shared tracker for cross-agent dedup
                    if isinstance(values, list):
                        for v in values:
                            key = f"{discovery_type}:{v}"
                            if key not in disc_tracker:
                                disc_tracker.add(key)
                                self.discoveries.add(key)  # Also track locally
                                total_discovery_bonus += self.DISCOVERY_BONUSES[discovery_type]
                    else:
                        key = f"{discovery_type}:{values}"
                        if key not in disc_tracker:
                            disc_tracker.add(key)
                            self.discoveries.add(key)
                            total_discovery_bonus += self.DISCOVERY_BONUSES[discovery_type]
            
            breakdown.discovery_bonus = total_discovery_bonus
            if total_discovery_bonus > 0:
                explanations.append(f"Discoveries: +{total_discovery_bonus:.1f}")
        
        # 4b. Phase 6.4: MS2-specific shaped reward bonus
        # Gives extra reward for targeting known MS2 vulnerable services
        if self._ms2_graph and new_discoveries:
            ms2_bonus = self._ms2_graph.get_shaped_reward(
                command=command,
                discoveries=new_discoveries,
                state_flags=state_flags,
            )
            if ms2_bonus > 0:
                breakdown.discovery_bonus += ms2_bonus
                explanations.append(f"MS2-bonus: +{ms2_bonus:.1f}")
        
        # 5. Phase advancement bonus
        new_phase = get_phase_from_state(state_flags)
        
        if self._phase_order(new_phase) > self._phase_order(self.highest_phase):
            # Calculate bonus based on how many phases advanced
            old_reward = self.PHASE_REWARDS.get(self.highest_phase, 0)
            new_reward = self.PHASE_REWARDS.get(new_phase, 0)
            bonus = (new_reward - old_reward) * self.phase_advance_multiplier
            
            breakdown.phase_advance_bonus = bonus
            explanations.append(
                f"Phase advance ({self.highest_phase.name} → {new_phase.name}): +{bonus:.1f}"
            )
            
            self.highest_phase = new_phase
        
        # 5b. Phase-appropriateness bonus (Phase 6.7)
        # Teaches PPO WHEN to use each command — the "reasoning" signal
        # +3.0 if command is appropriate for the current phase
        # -2.0 if command belongs to a completely wrong phase
        if template:
            cmd_phase_order = self._phase_order(template.phase)
            cur_phase_order = self._phase_order(current_phase)
            phase_distance = abs(cmd_phase_order - cur_phase_order)
            
            if phase_distance == 0:
                # Perfect phase match — this is the RIGHT time for this command
                breakdown.efficiency_bonus += 3.0
                explanations.append("🎯 Phase-match: +3.0")
            elif phase_distance == 1:
                # Adjacent phase — acceptable (e.g., ENUM tool during early EXPLOIT)
                breakdown.efficiency_bonus += 1.0
                explanations.append("Phase-adjacent: +1.0")
            elif phase_distance >= 3:
                # Way off — scanning during exfiltration is wasted effort
                breakdown.failure_penalty += 2.0
                explanations.append(f"⚠️ Wrong-phase ({template.phase.name} during {current_phase.name}): -2.0")
        
        # 6. Progress bonus - reward commands that change state (ADD to base, don't overwrite)
        new_flags = sum(1 for k, v in state_flags.items() if v and k not in self.discoveries)
        if new_flags > 0:
            breakdown.progress_bonus += new_flags * 0.5
            explanations.append(f"Progress ({new_flags} new flags): +{new_flags * 0.5:.1f}")
        
        # 7. Efficiency bonus - reward useful commands
        if success and (breakdown.discovery_bonus > 0 or breakdown.progress_bonus > 0):
            # Track that this was useful
            self.last_useful_command_idx = len(self.command_history)
            
            # Bonus based on recent efficiency
            recent_useful = sum(
                1 for i, r in enumerate(self.reward_history[-self.efficiency_window:])
                if r > 0
            )
            efficiency_rate = recent_useful / min(len(self.reward_history), self.efficiency_window) if self.reward_history else 0
            breakdown.efficiency_bonus = efficiency_rate * 1.0
            if breakdown.efficiency_bonus > 0:
                explanations.append(f"Efficiency: +{breakdown.efficiency_bonus:.1f}")
        
        # 7b. DIVERSITY BONUS - reward using different command prefixes
        # Industry standard: encourage broad exploration across tool categories
        cmd_prefix = command.strip().split()[0].lower() if command else ""
        used_prefixes = set(c.strip().split()[0].lower() for c in self.command_history if c.strip())
        if cmd_prefix and cmd_prefix not in used_prefixes:
            # First time using this tool type - bonus based on total diversity
            diversity_count = len(used_prefixes) + 1
            diversity_bonus = min(4.0, diversity_count * 0.5)  # Up to +4 for diverse toolkit
            breakdown.novelty_bonus += diversity_bonus
            explanations.append(f"🛠️ New tool ({diversity_count} unique): +{diversity_bonus:.1f}")
        
        # 8. Failure penalty
        if not success:
            # Graduated penalty based on command type
            if template:
                if template.phase in [AttackPhase.EXPLOITATION, AttackPhase.PRIVILEGE_ESCALATION]:
                    # Lower penalty for exploitation attempts (expected to fail often)
                    breakdown.failure_penalty = 0.5
                else:
                    breakdown.failure_penalty = 1.0
            else:
                breakdown.failure_penalty = 2.0  # Higher penalty for invalid commands
            
            explanations.append(f"Failure: -{breakdown.failure_penalty:.1f}")
        
        # Calculate total
        breakdown.calculate_total()
        breakdown.explanation = " | ".join(explanations)
        
        # Update tracking
        self.command_history.append(command)
        self.template_usage[template_name] += 1
        self.phase_history.append(current_phase)
        self.reward_history.append(breakdown.total)
        
        return breakdown
    
    def _phase_order(self, phase: AttackPhase) -> int:
        """Get numerical order of a phase."""
        order = {
            AttackPhase.RECON: 0,
            AttackPhase.ENUMERATION: 1,
            AttackPhase.EXPLOITATION: 2,
            AttackPhase.PRIVILEGE_ESCALATION: 3,
            AttackPhase.LATERAL_MOVEMENT: 4,
            AttackPhase.POST_EXPLOITATION: 5,
            AttackPhase.EXFILTRATION: 6,
            AttackPhase.CLOSEOUT: 7,
        }
        return order.get(phase, 0)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for current session."""
        total_reward = sum(self.reward_history)
        avg_reward = total_reward / len(self.reward_history) if self.reward_history else 0
        
        # Command efficiency
        useful_commands = sum(1 for r in self.reward_history if r > 1)
        efficiency = useful_commands / len(self.reward_history) if self.reward_history else 0
        
        # Phase distribution
        phase_counts = defaultdict(int)
        for phase in self.phase_history:
            phase_counts[phase.name] += 1
        
        return {
            "total_commands": len(self.command_history),
            "unique_templates": len(self.template_usage),
            "total_reward": total_reward,
            "average_reward": avg_reward,
            "efficiency": efficiency,
            "discoveries": len(self.discoveries),
            "highest_phase": self.highest_phase.name,
            "phase_distribution": dict(phase_counts),
            "duration_seconds": (datetime.now() - self.session_start).total_seconds()
        }
    
    def get_redundancy_report(self) -> List[Tuple[str, int, float]]:
        """
        Get report of most redundant commands.
        
        Returns:
            List of (template_name, count, penalty_accumulated) tuples
        """
        report = []
        for template, count in sorted(
            self.template_usage.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if count > 1:
                # Estimate penalty accumulated
                penalty = (count - 1) * self.max_redundancy_penalty * 0.5
                report.append((template, count, penalty))
        
        return report
    
    def suggest_exploration(
        self,
        current_phase: AttackPhase,
        state_flags: Dict[str, bool]
    ) -> List[str]:
        """
        Suggest commands that haven't been tried much.
        
        Args:
            current_phase: Current attack phase
            state_flags: Current state flags
            
        Returns:
            List of template names to try
        """
        suggestions = []
        
        # Get all valid commands
        from ..commands.command_registry import get_valid_commands_for_state
        
        valid = get_valid_commands_for_state(state_flags, current_phase)
        
        for cmd in valid:
            if self.template_usage.get(cmd.name, 0) < 2:
                suggestions.append(cmd.name)
        
        # Sort by typical reward
        suggestions.sort(
            key=lambda n: COMMAND_REGISTRY[n].typical_reward if n in COMMAND_REGISTRY else 0,
            reverse=True
        )
        
        return suggestions[:5]
    
    def is_stuck(self, window: int = 10, threshold: float = 0.5) -> bool:
        """
        Detect if agent is stuck (low recent rewards).
        
        Args:
            window: Number of recent commands to check
            threshold: Average reward threshold below which we're stuck
            
        Returns:
            True if stuck, False otherwise
        """
        if len(self.reward_history) < window:
            return False
        
        recent = self.reward_history[-window:]
        avg = sum(recent) / len(recent)
        
        return avg < threshold
    
    def get_phase_progress(self) -> Dict[str, Any]:
        """Get detailed phase progression info."""
        return {
            "current_phase": self.highest_phase.name,
            "phases_reached": list(set(p.name for p in self.phase_history)),
            "phase_timeline": [p.name for p in self.phase_history[-20:]],
            "phase_rewards": {
                p.name: self.PHASE_REWARDS[p]
                for p in AttackPhase
                if self._phase_order(p) <= self._phase_order(self.highest_phase)
            }
        }


def create_reward_calculator(
    novelty_weight: float = 1.0,
    redundancy_decay: float = 0.5,
    max_redundancy_penalty: float = 5.0,
    ms2_mode: bool = False,
) -> SmartRewardCalculator:
    """
    Factory function to create a SmartRewardCalculator.
    
    Args:
        novelty_weight: Weight for novelty bonus
        redundancy_decay: Decay rate for redundancy penalty
        max_redundancy_penalty: Maximum redundancy penalty
        ms2_mode: Enable Metasploitable 2 reward shaping
        
    Returns:
        Configured SmartRewardCalculator instance
    """
    return SmartRewardCalculator(
        novelty_weight=novelty_weight,
        redundancy_decay=redundancy_decay,
        max_redundancy_penalty=max_redundancy_penalty,
        ms2_mode=ms2_mode,
    )
