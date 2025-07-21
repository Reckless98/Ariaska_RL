# core/interfaces/agent_interface.py — ARIASKA Agent Interface v2.1 APEX
# 🧠 Core Interface | 🔄 Multi-Agent Protocol | 📦 Base Requirements
from typing import Dict, List, Any, Optional, Union, Tuple

class AgentInterface:
    """Base interface for all ARIASKA agents."""
    
    @property
    def agent_id(self) -> str:
        """Return the agent's unique identifier."""
        raise NotImplementedError("Agent must implement agent_id property")
    
    @property
    def role(self) -> str:
        """Return the agent's role in the multi-agent system."""
        raise NotImplementedError("Agent must implement role property")
    
    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the next action based on current state."""
        raise NotImplementedError("Agent must implement act method")

    def learn(self, state: Dict[str, Any], action: Dict[str, Any], 
             reward: float, next_state: Dict[str, Any], done: bool) -> float:
        """Update agent's knowledge based on experience."""
        raise NotImplementedError("Agent must implement learn method")
    
    def process_directive(self, directive_type: str, parameters: Dict[str, Any], 
                         priority: int = 1, source_agent: str = "OrionAgent") -> Dict[str, Any]:
        """Process a strategic directive from another agent."""
        agent_id = getattr(self, 'agent_id', 'Agent')
        print(f"{agent_id} received directive {directive_type} from {source_agent} (priority {priority})")
        return {"processed": False, "reason": "Not implemented by this agent"}

    def sync_memory(self) -> bool:
        """Synchronize agent memory with central memory system."""
        return False

    def reset(self) -> None:
        """Reset agent state for a new episode."""
        pass
    
    def simulate_step(self, episode: int = 1, step: int = 1,
                     shared_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Simulate a step for this agent in the training loop."""
        return {
            "agent_id": getattr(self, 'agent_id', 'Unknown'),
            "episode": episode,
            "step": step,
            "command": "default_action",
            "reward": 0.0,
            "done": False
        }
    
    def generate_hint(self) -> str:
        """Generate a hint or suggestion for the current state."""
        return f"{getattr(self, 'agent_id', 'Agent')} suggests analyzing the current environment"
    
    def provide_strategic_insights(self) -> Dict[str, Any]:
        """Provide strategic insights for other agents."""
        return {
            "agent_id": getattr(self, 'agent_id', 'Unknown'),
            "insights": "Standard operation mode",
            "recommendations": ["Continue current strategy"],
            "confidence": 0.5
        }
    
    def execute_command(self, command: str) -> tuple:
        """Execute a command and return results."""
        agent_id = getattr(self, 'agent_id', 'Agent')
        return (command, f"{agent_id} executed: {command}", 0)
    
    def get_action(self, state: Dict[str, Any], **kwargs) -> str:
        """Get the next action based on current state."""
        return "analyze_environment"
    
    def share_knowledge(self, target_agent) -> bool:
        """Share knowledge with another agent."""
        agent_id = getattr(self, 'agent_id', 'Agent')
        if hasattr(target_agent, 'agent_id'):
            print(f"{agent_id} sharing knowledge with {target_agent.agent_id}")
        return True
