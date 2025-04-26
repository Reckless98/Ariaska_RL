# core/interfaces/agent_interface.py — ARIASKA Agent Interface v2.0 APEX
# 🧠 Core Interface | 🔄 Multi-Agent Protocol | 📦 Base Requirements
from typing import Dict, List, Any, Optional, Union, Tuple

class AgentInterface:
    """
    Base interface for all ARIASKA agents.
    
    Defines core methods that all agents must implement:
    - Basic agent functionality (act, learn, reset)
    - Hierarchical directive processing
    - Memory synchronization
    - Simulation step handling
    """
    
    @property
    def agent_id(self) -> str:
        """Return the agent's unique identifier."""
        raise NotImplementedError("Agent must implement agent_id property")
    
    @property
    def role(self) -> str:
        """Return the agent's role in the multi-agent system."""
        raise NotImplementedError("Agent must implement role property")
    
    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the next action based on current state.
        
        Args:
            state: Current environment state
            
        Returns:
            Dict with action information
        """
        raise NotImplementedError("Agent must implement act method")

    def learn(self, state: Dict[str, Any], action: Dict[str, Any], 
             reward: float, next_state: Dict[str, Any], done: bool) -> float:
        """
        Update agent's knowledge based on experience.
        
        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode is complete
            
        Returns:
            Loss value from learning
        """
        raise NotImplementedError("Agent must implement learn method")
    
    def process_directive(self, directive_type: str, parameters: Dict[str, Any], 
                         priority: int = 1, source_agent: str = "OrionAgent") -> Dict[str, Any]:
        """
        Process a strategic directive from another agent (typically OrionAgent).
        
        Args:
            directive_type: Type of directive (e.g., STEALTH, AGGRESSIVE)
            parameters: Additional parameters for the directive
            priority: Priority level (1-5, with 5 being highest)
            source_agent: Agent that issued the directive
            
        Returns:
            Dict with processing results
        """
        # Default implementation: Log directive but take no action
        print(f"{self.agent_id} received directive {directive_type} from {source_agent} (priority {priority})")
        return {"processed": False, "reason": "Not implemented by this agent"}

    def sync_memory(self) -> bool:
        """
        Synchronize agent memory with central memory system.
        
        Returns:
            True if synchronization was successful, False otherwise
        """
        return False  # Default implementation
    
    def reset(self) -> None:
        """Reset agent state for a new episode."""
        pass  # Default implementation does nothing
    
    def simulate_step(self, episode: int = 1, step: int = 1,
                     shared_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Simulate a step for this agent in the training loop.
        
        Args:
            episode: Current episode number
            step: Current step number
            shared_context: Shared context from agent manager
            
        Returns:
            Dict with step results
        """
        raise NotImplementedError("Agent must implement simulate_step method")
    
    def get_base_commands(self) -> List[str]:
        """
        Get the list of base commands this agent can execute.
        
        Returns:
            List of command strings
        """
        return []  # Default implementation returns empty list
    
    def receive_global_context(self, context: Dict[str, Any]) -> None:
        """
        Receive global context information from the agent manager or environment.
        
        Args:
            context: Global context information
        """
        pass  # Default implementation does nothing
