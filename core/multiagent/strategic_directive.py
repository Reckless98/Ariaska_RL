"""
strategic_directive.py — Strategic Directive System for Multi-Agent Coordination
Implements a formalized directive system that enables hierarchical coordination with OrionAgent as strategic overseer.
Features:
- Structured directives with priority and TTL
- Directive status tracking and updates
- Agent-specific directive filtering
- Directive chain formation and execution
"""

import time
import uuid
import random
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from enum import Enum, auto

# Logger setup
logger = logging.getLogger(__name__)


class DirectiveStatus(Enum):
    """Status enum for strategic directives"""
    ISSUED = auto()      # Newly issued directive
    RECEIVED = auto()    # Directive received by target agent
    IN_PROGRESS = auto() # Directive currently being processed
    COMPLETED = auto()   # Directive successfully completed
    REJECTED = auto()    # Directive rejected by agent
    EXPIRED = auto()     # Directive timed out
    SUPERSEDED = auto()  # Directive superseded by newer directive


class DirectiveType(Enum):
    """Types of strategic directives that can be issued"""
    FOCUS_TARGET = auto()       # Focus on specific target
    INCREASE_STEALTH = auto()   # Increase stealth level
    DECREASE_STEALTH = auto()   # Decrease stealth level for faster operation
    ADAPTIVE_DEFENSE = auto()   # Adjust defense posture
    SCOUT_AREA = auto()         # Request scouting of specific area
    GATHER_INTEL = auto()       # Gather intelligence
    EXECUTE_EXPLOIT = auto()    # Execute specific exploit
    DEPLOY_COUNTERMEASURE = auto()  # Deploy specific countermeasure
    BACKUP_AGENT = auto()       # Request backup from another agent
    RETREAT = auto()            # Retreat from current action
    CHANGE_STRATEGY = auto()    # Change overall strategy
    COORDINATE_ACTION = auto()  # Coordinate specific action with other agent
    HALT_OPERATION = auto()     # Halt current operation
    RESUME_OPERATION = auto()   # Resume halted operation
    CUSTOM = auto()             # Custom directive type


class StrategicDirective:
    """
    Represents a strategic directive issued by one agent (typically OrionAgent)
    to another agent or group of agents in the ARIASKA_RL system.
    
    Strategic directives enable hierarchical coordination with OrionAgent as the
    strategic overseer for the multi-agent system.
    """
    
    def __init__(
        self, 
        directive_type: Union[DirectiveType, str],
        target_agent: str,
        source_agent: str = "OrionAgent",
        parameters: Dict[str, Any] = None,
        priority: int = 1,
        ttl: int = 10,
        step: int = 0,
        episode: int = 0
    ):
        """
        Initialize a new strategic directive.
        
        Args:
            directive_type: Type of directive (from DirectiveType enum or string)
            target_agent: Agent ID that should execute this directive
            source_agent: Agent ID issuing the directive (typically OrionAgent)
            parameters: Additional parameters for the directive
            priority: Priority level (1-5, with 5 being highest)
            ttl: Time-to-live in steps before the directive expires
            step: Current step number when directive was created
            episode: Current episode number when directive was created
        """
        self.id = str(uuid.uuid4())
        
        # Convert string to enum if needed
        if isinstance(directive_type, str):
            try:
                self.directive_type = DirectiveType[directive_type.upper()]
            except KeyError:
                self.directive_type = DirectiveType.CUSTOM
                self.custom_type = directive_type
        else:
            self.directive_type = directive_type
            self.custom_type = None
            
        self.target_agent = target_agent
        self.source_agent = source_agent
        self.parameters = parameters or {}
        self.priority = max(1, min(5, priority))  # Clamp between 1-5
        self.ttl = ttl
        self.step = step
        self.episode = episode
        self.timestamp = time.time()
        self.status = DirectiveStatus.ISSUED
        self.result = None
        self.expiration = step + ttl
    
    def get_type_name(self) -> str:
        """Get the string name of the directive type"""
        if self.directive_type == DirectiveType.CUSTOM:
            return self.custom_type
        return self.directive_type.name
    
    def update_status(self, new_status: DirectiveStatus, result: Any = None) -> None:
        """
        Update the status of this directive.
        
        Args:
            new_status: New status for the directive
            result: Optional result from processing the directive
        """
        self.status = new_status
        if result is not None:
            self.result = result
            
    def is_expired(self, current_step: int) -> bool:
        """
        Check if this directive has expired.
        
        Args:
            current_step: Current step number
            
        Returns:
            True if directive has expired, False otherwise
        """
        return current_step > self.expiration
    
    def is_active(self) -> bool:
        """
        Check if this directive is still active.
        
        Returns:
            True if directive is active, False otherwise
        """
        return self.status in [
            DirectiveStatus.ISSUED,
            DirectiveStatus.RECEIVED,
            DirectiveStatus.IN_PROGRESS
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert directive to dictionary format.
        
        Returns:
            Dictionary representation of directive
        """
        return {
            "id": self.id,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "directive_type": self.get_type_name(),
            "parameters": self.parameters,
            "priority": self.priority,
            "status": self.status.name,
            "timestamp": self.timestamp,
            "step": self.step,
            "episode": self.episode,
            "ttl": self.ttl,
            "expiration": self.expiration,
            "result": self.result
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategicDirective':
        """
        Create directive from dictionary format.
        
        Args:
            data: Dictionary representation of directive
            
        Returns:
            New StrategicDirective instance
        """
        directive = cls(
            directive_type=data["directive_type"],
            target_agent=data["target_agent"],
            source_agent=data["source_agent"],
            parameters=data["parameters"],
            priority=data["priority"],
            ttl=data["ttl"],
            step=data["step"],
            episode=data["episode"]
        )
        
        # Restore other properties
        directive.id = data["id"]
        directive.timestamp = data["timestamp"]
        directive.status = DirectiveStatus[data["status"]]
        directive.expiration = data["expiration"]
        directive.result = data["result"]
        
        return directive
        
    def __str__(self):
        return f"[{self.priority}] {self.get_type_name()} -> {self.target_agent} ({self.status.name})"
    
    def __repr__(self):
        return f"StrategicDirective({self.get_type_name()}, {self.target_agent}, {self.priority}, {self.status.name})"


class DirectiveManager:
    """
    Manages strategic directives in the ARIASKA_RL system.
    
    This class is responsible for:
    1. Tracking all issued directives
    2. Updating directive statuses
    3. Expiring old directives
    4. Providing directives to agents
    """
    
    def __init__(self, memory_router=None):
        """
        Initialize a new directive manager.
        
        Args:
            memory_router: Optional memory router for persistence
        """
        self.directives = {}  # Map of directive_id -> directive
        self.memory_router = memory_router
        
        # Indices for efficient lookup
        self.directives_by_target = {}
        self.directives_by_source = {}
        self.directives_by_type = {}
        self.directives_by_status = {}
    
    def issue_directive(
        self,
        directive_type: Union[DirectiveType, str],
        target_agent: str,
        source_agent: str = "OrionAgent",
        parameters: Dict[str, Any] = None,
        priority: int = 1,
        ttl: int = 10,
        step: int = 0,
        episode: int = 0
    ) -> StrategicDirective:
        """
        Issue a new strategic directive.
        
        Args:
            directive_type: Type of directive
            target_agent: Agent that should execute this directive
            source_agent: Agent issuing the directive (typically OrionAgent)
            parameters: Additional parameters for the directive
            priority: Priority level (1-5)
            ttl: Time-to-live in steps before the directive expires
            step: Current step number
            episode: Current episode number
            
        Returns:
            Newly created directive
        """
        directive = StrategicDirective(
            directive_type=directive_type,
            target_agent=target_agent,
            source_agent=source_agent,
            parameters=parameters,
            priority=priority,
            ttl=ttl,
            step=step,
            episode=episode
        )
        
        # Store directive
        self.directives[directive.id] = directive
        
        # Update indices
        self._add_to_index(self.directives_by_target, target_agent, directive)
        self._add_to_index(self.directives_by_source, source_agent, directive)
        self._add_to_index(self.directives_by_type, directive.get_type_name(), directive)
        self._add_to_index(self.directives_by_status, directive.status, directive)
        
        # Log to memory router if available
        if self.memory_router:
            self.memory_router.log_directive(
                source_agent=source_agent,
                target_agent=target_agent,
                directive_type=directive.get_type_name(),
                parameters=parameters or {},
                priority=priority,
                step=step,
                episode=episode,
                ttl=ttl
            )
        
        logger.info(f"Directive issued: {directive}")
        return directive
    
    def update_directive(
        self,
        directive_id: str,
        status: DirectiveStatus,
        result: Any = None
    ) -> Optional[StrategicDirective]:
        """
        Update the status of a directive.
        
        Args:
            directive_id: ID of the directive to update
            status: New status for the directive
            result: Optional result from processing the directive
            
        Returns:
            Updated directive, or None if not found
        """
        if directive_id not in self.directives:
            logger.warning(f"Cannot update directive {directive_id}: not found")
            return None
            
        directive = self.directives[directive_id]
        
        # Remove from old status index
        self._remove_from_index(self.directives_by_status, directive.status, directive)
        
        # Update directive
        old_status = directive.status
        directive.update_status(status, result)
        
        # Add to new status index
        self._add_to_index(self.directives_by_status, directive.status, directive)
        
        # Log to memory router if available
        if self.memory_router:
            self.memory_router.update_directive_status(
                directive_id=directive_id,
                status=status.name,
                result=result
            )
        
        logger.info(f"Directive {directive_id} updated: {old_status.name} -> {status.name}")
        return directive
    
    def get_directive(self, directive_id: str) -> Optional[StrategicDirective]:
        """
        Get a directive by ID.
        
        Args:
            directive_id: ID of the directive to get
            
        Returns:
            Directive if found, None otherwise
        """
        return self.directives.get(directive_id)
    
    def get_agent_directives(
        self,
        agent_id: str,
        active_only: bool = True,
        as_target: bool = True,
        limit: int = 10
    ) -> List[StrategicDirective]:
        """
        Get directives for a specific agent.
        
        Args:
            agent_id: ID of the agent
            active_only: Only return active directives
            as_target: If True, get directives where agent is target; 
                      If False, get directives issued by the agent
            limit: Maximum number of directives to return
            
        Returns:
            List of directives
        """
        # Use memory router if available
        if self.memory_router:
            directive_dicts = self.memory_router.get_agent_directives(
                agent_id=agent_id,
                active_only=active_only,
                as_target=as_target,
                limit=limit
            )
            
            directives = []
            for d_dict in directive_dicts:
                # Try to find in local cache first
                directive = self.directives.get(d_dict["id"])
                if not directive:
                    # Create new directive from dict
                    try:
                        directive = StrategicDirective(
                            directive_type=d_dict["directive_type"],
                            target_agent=d_dict["target_agent"],
                            source_agent=d_dict["source_agent"],
                            parameters=d_dict["parameters"],
                            priority=d_dict["priority"],
                            ttl=d_dict.get("ttl", 10),
                            step=d_dict.get("step", 0),
                            episode=d_dict.get("episode", 0)
                        )
                        directive.id = d_dict["id"]
                        directive.status = DirectiveStatus[d_dict["status"].upper()]
                    except Exception as e:
                        logger.error(f"Error creating directive from dict: {e}")
                        continue
                        
                directives.append(directive)
                
            return directives
        
        # Use local indices
        index = self.directives_by_target if as_target else self.directives_by_source
        if agent_id not in index:
            return []
            
        directives = list(index[agent_id])
        
        # Filter active directives if needed
        if active_only:
            directives = [d for d in directives if d.is_active()]
            
        # Sort by priority (desc) and timestamp (asc)
        directives.sort(key=lambda d: (-d.priority, d.timestamp))
        
        return directives[:limit]
    
    def get_directives_by_type(
        self,
        directive_type: Union[DirectiveType, str],
        active_only: bool = False,
        limit: int = 10
    ) -> List[StrategicDirective]:
        """
        Get directives of a specific type.
        
        Args:
            directive_type: Type of directives to get
            active_only: Only return active directives
            limit: Maximum number of directives to return
            
        Returns:
            List of directives
        """
        # Convert to string if needed
        if isinstance(directive_type, DirectiveType):
            type_name = directive_type.name
        else:
            type_name = directive_type
            
        if type_name not in self.directives_by_type:
            return []
            
        directives = list(self.directives_by_type[type_name])
        
        # Filter active directives if needed
        if active_only:
            directives = [d for d in directives if d.is_active()]
            
        # Sort by priority (desc) and timestamp (asc)
        directives.sort(key=lambda d: (-d.priority, d.timestamp))
        
        return directives[:limit]
    
    def expire_directives(self, current_step: int) -> List[StrategicDirective]:
        """
        Expire directives that have reached their TTL.
        
        Args:
            current_step: Current step number
            
        Returns:
            List of expired directives
        """
        expired = []
        
        for directive in list(self.directives.values()):
            if directive.is_active() and directive.is_expired(current_step):
                self.update_directive(directive.id, DirectiveStatus.EXPIRED)
                expired.append(directive)
                
        return expired
    
    def clear_directives(self, agent_id: str = None, completed_only: bool = True) -> int:
        """
        Clear directives from the manager.
        
        Args:
            agent_id: Optional agent ID to clear directives for
            completed_only: Only clear completed/expired directives
            
        Returns:
            Number of directives cleared
        """
        to_remove = []
        
        for directive_id, directive in self.directives.items():
            if agent_id and directive.target_agent != agent_id:
                continue
                
            if completed_only and directive.is_active():
                continue
                
            to_remove.append(directive_id)
            
        # Remove directives
        for directive_id in to_remove:
            directive = self.directives[directive_id]
            
            # Remove from all indices
            self._remove_from_index(self.directives_by_target, directive.target_agent, directive)
            self._remove_from_index(self.directives_by_source, directive.source_agent, directive)
            self._remove_from_index(self.directives_by_type, directive.get_type_name(), directive)
            self._remove_from_index(self.directives_by_status, directive.status, directive)
            
            # Remove from directives dict
            del self.directives[directive_id]
            
        return len(to_remove)
    
    def get_directive_stats(self) -> Dict[str, Any]:
        """
        Get statistics about directives.
        
        Returns:
            Dictionary with directive statistics
        """
        stats = {
            "total_directives": len(self.directives),
            "active_directives": sum(1 for d in self.directives.values() if d.is_active()),
            "by_status": {},
            "by_agent": {},
            "by_source": {},
            "by_type": {},
            "by_priority": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        }
        
        # Count by status
        for status in DirectiveStatus:
            stats["by_status"][status.name] = len(self.directives_by_status.get(status, set()))
            
        # Count by target agent
        for agent, directives in self.directives_by_target.items():
            stats["by_agent"][agent] = len(directives)
            
        # Count by source agent
        for agent, directives in self.directives_by_source.items():
            stats["by_source"][agent] = len(directives)
            
        # Count by type
        for type_name, directives in self.directives_by_type.items():
            stats["by_type"][type_name] = len(directives)
            
        # Count by priority
        for directive in self.directives.values():
            stats["by_priority"][directive.priority] += 1
            
        return stats
    
    def _add_to_index(self, index, key, directive):
        """Helper method to add directive to an index"""
        if key not in index:
            index[key] = set()
        index[key].add(directive)
    
    def _remove_from_index(self, index, key, directive):
        """Helper method to remove directive from an index"""
        if key in index and directive in index[key]:
            index[key].remove(directive)
            if not index[key]:
                del index[key]


class DirectiveChain:
    """
    Represents a chain of strategic directives to be executed in sequence.
    
    This enables more complex multi-step coordination between agents.
    """
    
    def __init__(self, name: str = None, agent_id: str = "OrionAgent"):
        """
        Initialize a new directive chain.
        
        Args:
            name: Optional name for the chain
            agent_id: Agent creating this chain
        """
        self.id = str(uuid.uuid4())
        self.name = name or f"Chain-{self.id[:8]}"
        self.agent_id = agent_id
        self.directives = []
        self.current_index = 0
        self.created_at = time.time()
        self.completed = False
        
    def add_directive(self, directive: StrategicDirective) -> None:
        """
        Add a directive to the chain.
        
        Args:
            directive: Directive to add
        """
        self.directives.append(directive)
        
    def get_current_directive(self) -> Optional[StrategicDirective]:
        """
        Get the current directive in the chain.
        
        Returns:
            Current directive, or None if chain is empty or completed
        """
        if self.completed or self.current_index >= len(self.directives):
            return None
            
        return self.directives[self.current_index]
        
    def advance(self) -> bool:
        """
        Advance to the next directive in the chain.
        
        Returns:
            True if there is a next directive, False if chain is completed
        """
        self.current_index += 1
        
        if self.current_index >= len(self.directives):
            self.completed = True
            return False
            
        return True
        
    def is_completed(self) -> bool:
        """
        Check if the chain is completed.
        
        Returns:
            True if chain is completed, False otherwise
        """
        return self.completed
        
    def reset(self) -> None:
        """Reset the chain to the first directive"""
        self.current_index = 0
        self.completed = False
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert chain to dictionary format.
        
        Returns:
            Dictionary representation of chain
        """
        return {
            "id": self.id,
            "name": self.name,
            "agent_id": self.agent_id,
            "directives": [d.to_dict() for d in self.directives],
            "current_index": self.current_index,
            "created_at": self.created_at,
            "completed": self.completed
        }

# Global directive manager instance for system-wide use
directive_manager = DirectiveManager()

# For backward compatibility with code that might expect add_directive
setattr(directive_manager, 'add_directive', directive_manager.issue_directive)