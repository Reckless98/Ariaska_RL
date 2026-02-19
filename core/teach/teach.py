import os
import json
import threading
import asyncio
from typing import Dict, Any, Optional, List
from rich.console import Console
from core.utils.llm_router import LLMRouter
from core.utils.memory_manager import MemoryManager

console = Console()

class TemplateEngine:
    """Handles command templating and parsing for teaching events."""
    def parse_action(self, command: str, phase: str, reward: float, **kwargs) -> Dict[str, Any]:
        # Standardize action object
        return {
            "command": command,
            "phase": phase,
            "reward": reward,
            "description": kwargs.get("description", ""),
            "when": kwargs.get("when", ""),
            "why": kwargs.get("why", ""),
            "full_command": kwargs.get("full_command", command),
            "meta": kwargs.get("meta", {}),
        }

class TeacherLogger:
    """Handles logging of teaching events and batching file I/O."""
    def __init__(self, log_path="logs/teach_events.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._lock = threading.Lock()
        self._buffer = []
        self._flush_interval = 10  # flush every 10 events
        self._event_count = 0

    def log_event(self, event: Dict[str, Any]):
        with self._lock:
            self._buffer.append(event)
            self._event_count += 1
            if self._event_count % self._flush_interval == 0:
                self.flush()

    def flush(self):
        with self._lock:
            if not self._buffer:
                return
            with open(self.log_path, "a") as f:
                for event in self._buffer:
                    f.write(json.dumps(event) + "\n")
            self._buffer.clear()

    def summarize(self, n=10) -> List[Dict[str, Any]]:
        """Return the last n logged events."""
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, "r") as f:
            lines = f.readlines()[-n:]
        return [json.loads(line) for line in lines]

class TeacherPolicy:
    """Decides when to teach and how to prioritize taught actions."""
    def should_teach(self, action: Dict[str, Any], memory: List[Dict[str, Any]]) -> bool:
        # Avoid teaching exact or semantically redundant actions
        commands = [a["command"] for a in memory]
        if action["command"] in commands:
            return False
        # TODO: Add semantic redundancy check if needed
        return True

class TeachModule:
    """
    Refactored TeachModule:
    - Separation of concerns (templating, logging, policy)
    - Async LLM routing via LLMRouter
    - Thread-safe caching and batching
    - Clean API for agent/trainer integration
    """
    def __init__(self, agent_name="RedAgent", memory_manager: Optional[MemoryManager]=None):
        self.agent_name = agent_name
        self.memory_manager = memory_manager or MemoryManager(agent_name)
        self.template_engine = TemplateEngine()
        self.logger = TeacherLogger(log_path=f"logs/{agent_name}_teach_events.jsonl")
        self.policy = TeacherPolicy()
        self.llm_router = LLMRouter()
        self._cache = {}
        self._cache_lock = threading.Lock()

    def add_action(self, command: str, phase: str, reward: float, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Add a new taught action, with templating, deduplication, and logging.
        Returns standardized action object if added, else None.
        """
        action = self.template_engine.parse_action(command, phase, reward, **kwargs)
        memory = self.memory_manager.get_actions()
        if not self.policy.should_teach(action, memory):
            return None
        self.memory_manager.add_action(action)
        self.logger.log_event(action)
        return action

    async def summarize_action(self, action: Dict[str, Any]) -> str:
        """
        Use LLMRouter to summarize an action asynchronously.
        """
        prompt = f"Summarize the following cybersecurity action for teaching:\n{json.dumps(action)}"
        cache_key = f"summarize_{hash(json.dumps(action))}"
        with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
        # Use LLMRouter for async LLM call
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: self.llm_router.route_task("teach_action_parsing", prompt, model="gpt-5.2-codex")
        )
        with self._cache_lock:
            self._cache[cache_key] = response
        return response

    def add_action_sync(self, command: str, phase: str, reward: float, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Synchronous wrapper for add_action (for legacy code).
        """
        return self.add_action(command, phase, reward, **kwargs)

    def flush_logs(self):
        self.logger.flush()

    def get_recent_actions(self, n=10) -> List[Dict[str, Any]]:
        return self.logger.summarize(n=n)

    def register_taught_action_for_dqn(self, action: Dict[str, Any], replay_buffer):
        """
        Optionally add taught action to replay buffer for prioritized experience replay.
        """
        experience = {
            "state": action.get("meta", {}).get("state", [0.0]*512),
            "action": action["command"],
            "reward": action["reward"],
            "next_state": action.get("meta", {}).get("next_state", [0.0]*512),
            "gpt_tokens": action.get("meta", {}).get("gpt_tokens", 0)
        }
        replay_buffer.add(experience)

    def shutdown(self):
        self.flush_logs()

# Example usage:
# teach = TeachModule(agent_name="RedAgent")
# teach.add_action("nmap -sV 10.10.10.10", phase="recon", reward=2.0, description="Scan for open services")
# asyncio.run(teach.summarize_action({...}))
