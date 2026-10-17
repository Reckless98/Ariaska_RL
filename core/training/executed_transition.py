"""The sole observed action/outcome record; proposals are not experiences."""
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ExecutedTransition:
    role: str
    episode: int
    step: int
    state: Any
    next_state: Any
    action: int  # -1 means a command outside the role's fixed action map
    command: str
    template: str
    reward: float
    terminated: bool
    truncated: bool
    source: str
    policy_sample: Optional[dict] = None
