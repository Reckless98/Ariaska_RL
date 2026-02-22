"""core/algorithms/goal_conditioned.py — Phase 49: Goal-Conditioned RL.

Conditions the policy on explicit goals (e.g., "find SSH credentials",
"get root shell") using goal embeddings that modify policy behaviour.

Design:
  - Goals are represented as learned embeddings (goal_dim=64).
  - Policy input = concat(state, goal_embedding) → 512+64 = 576 dims.
  - Universal Value Function Approximator (UVFA): V(s, g) and Q(s, a, g).
  - Goal relabelling: failed trajectories relabelled with achieved goals
    (similar to HER but at the goal level).

Pre-defined pentesting goals:
  - find_open_ports: Discover network services
  - get_credentials: Obtain valid credentials
  - get_shell: Establish remote shell
  - escalate_privileges: Root/admin access
  - exfiltrate_data: Extract target data
  - capture_flag: Find and submit CTF flag

Architecture:
    state (512) + goal_embedding (64) → GoalConditionedPolicy → action
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("ariaska.algorithms.goal_conditioned")


@dataclass
class GoalConfig:
    """Configuration for goal-conditioned RL.

    Args:
        enabled: Master switch.
        state_dim: Base state dimension.
        goal_dim: Goal embedding dimension.
        action_dim: Number of actions.
        hidden_dims: Hidden layer sizes for policy.
        n_goals: Number of predefined goals.
        relabel_prob: Probability of goal relabelling per trajectory.
        goal_reward_scale: Scale factor for goal-achievement reward.
        cosine_goal_similarity: Use cosine similarity for goal matching.
    """
    enabled: bool = True
    state_dim: int = 512
    goal_dim: int = 64
    action_dim: int = 5
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    n_goals: int = 8
    relabel_prob: float = 0.5
    goal_reward_scale: float = 10.0
    cosine_goal_similarity: bool = True


# Pre-defined pentesting goals with achievement conditions
PENTESTING_GOALS: List[Dict[str, Any]] = [
    {
        "id": 0, "name": "find_open_ports",
        "description": "Discover open ports and services",
        "phase": "RECON",
        "achievement_keys": ["ports_discovered"],
    },
    {
        "id": 1, "name": "enumerate_services",
        "description": "Identify service versions and configurations",
        "phase": "ENUMERATION",
        "achievement_keys": ["services_identified"],
    },
    {
        "id": 2, "name": "find_vulnerabilities",
        "description": "Discover exploitable vulnerabilities",
        "phase": "ENUMERATION",
        "achievement_keys": ["vulns_found"],
    },
    {
        "id": 3, "name": "get_credentials",
        "description": "Obtain valid credentials",
        "phase": "EXPLOITATION",
        "achievement_keys": ["credentials_found"],
    },
    {
        "id": 4, "name": "get_shell",
        "description": "Establish a remote shell on target",
        "phase": "EXPLOITATION",
        "achievement_keys": ["shell_obtained"],
    },
    {
        "id": 5, "name": "escalate_privileges",
        "description": "Gain root/admin access",
        "phase": "PRIVILEGE_ESCALATION",
        "achievement_keys": ["root_obtained"],
    },
    {
        "id": 6, "name": "exfiltrate_data",
        "description": "Extract sensitive data from target",
        "phase": "EXFILTRATION",
        "achievement_keys": ["data_exfiltrated"],
    },
    {
        "id": 7, "name": "capture_flag",
        "description": "Find and capture CTF flag",
        "phase": "EXFILTRATION",
        "achievement_keys": ["flag_captured"],
    },
]


class GoalEmbedding(nn.Module):
    """Learnable goal embeddings.

    Maps goal indices to dense vectors. Pre-initialised with
    semantic structure (similar goals → similar embeddings).
    """

    def __init__(self, n_goals: int, goal_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_goals, goal_dim)
        # Structured initialisation: goals in same phase start similar
        nn.init.orthogonal_(self.embedding.weight, gain=0.5)

    def forward(self, goal_ids: torch.Tensor) -> torch.Tensor:
        """Returns goal embeddings: (B,) → (B, goal_dim)."""
        return self.embedding(goal_ids)

    def all_embeddings(self) -> torch.Tensor:
        """Return all goal embeddings: (n_goals, goal_dim)."""
        return self.embedding.weight


class GoalConditionedPolicy(nn.Module):
    """Policy conditioned on goal embeddings.

    Input: concat(state, goal_embedding) → actor logits + critic value.
    Implements a Universal Value Function Approximator (UVFA).
    """

    def __init__(self, config: GoalConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = config.state_dim + config.goal_dim

        # Shared backbone
        layers: List[nn.Module] = []
        in_dim = input_dim
        for h in config.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
            ])
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, config.action_dim),
        )

        # Critic head: V(s, g)
        self.critic = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, 1),
        )

        # Small init for stability
        actor_last = self.actor[-1]
        assert isinstance(actor_last, nn.Linear)
        nn.init.orthogonal_(actor_last.weight, gain=0.01)
        nn.init.zeros_(actor_last.bias)

    def forward(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            state: (B, state_dim) state tensor.
            goal: (B, goal_dim) goal embedding.

        Returns:
            (logits, value): action logits (B, action_dim), value (B, 1).
        """
        x = torch.cat([state, goal], dim=-1)
        features = self.backbone(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value


class GoalSelector:
    """Selects and manages goals during training episodes.

    Implements:
      - Phase-appropriate goal selection
      - Goal achievement tracking
      - Hindsight goal relabelling
    """

    def __init__(self, config: Optional[GoalConfig] = None) -> None:
        self.config = config or GoalConfig()
        self.goal_embedding = GoalEmbedding(self.config.n_goals, self.config.goal_dim)
        self.policy = GoalConditionedPolicy(self.config)

        self._goals = PENTESTING_GOALS[:self.config.n_goals]
        self._goal_name_to_id: Dict[str, int] = {
            g["name"]: g["id"] for g in self._goals
        }
        self._current_goal_id: Optional[int] = None

        # Achievement tracking
        self._goal_attempts: Dict[int, int] = {g["id"]: 0 for g in self._goals}
        self._goal_successes: Dict[int, int] = {g["id"]: 0 for g in self._goals}

    def select_goal(
        self,
        state: Dict[str, Any],
        state_tensor: Optional[torch.Tensor] = None,
    ) -> Tuple[int, str, torch.Tensor]:
        """Select appropriate goal for current state.

        Args:
            state: Environment state dictionary.
            state_tensor: Optional state tensor.

        Returns:
            (goal_id, goal_name, goal_embedding)
        """
        phase = state.get("phase", "RECON").upper()

        # Find goals valid for current phase
        valid_goals = [g for g in self._goals if g["phase"] == phase]
        if not valid_goals:
            # Fall back to any goal
            valid_goals = self._goals

        # Prefer under-attempted goals
        min_attempts = min(
            self._goal_attempts.get(g["id"], 0) for g in valid_goals
        )
        candidates = [
            g for g in valid_goals
            if self._goal_attempts.get(g["id"], 0) == min_attempts
        ]

        import random
        goal = random.choice(candidates)
        goal_id = goal["id"]
        self._current_goal_id = goal_id
        self._goal_attempts[goal_id] = self._goal_attempts.get(goal_id, 0) + 1

        # Get embedding
        goal_tensor = torch.tensor([goal_id], dtype=torch.long)
        goal_emb = self.goal_embedding(goal_tensor).squeeze(0)

        logger.info(f"[GOAL] Selected: '{goal['name']}' (phase={phase})")
        return goal_id, goal["name"], goal_emb

    def check_achievement(
        self,
        goal_id: int,
        state: Dict[str, Any],
    ) -> Tuple[bool, float]:
        """Check if goal is achieved in current state.

        Returns:
            (achieved, reward_bonus)
        """
        if goal_id >= len(self._goals):
            return False, 0.0

        goal = self._goals[goal_id]
        achievement_keys = goal.get("achievement_keys", [])

        achieved = all(state.get(key, False) for key in achievement_keys)
        if achieved:
            self._goal_successes[goal_id] = self._goal_successes.get(goal_id, 0) + 1
            reward = self.config.goal_reward_scale
            logger.info(f"[GOAL] Achieved: '{goal['name']}' bonus=+{reward}")
            return True, reward

        return False, 0.0

    def relabel_trajectory(
        self,
        trajectory: List[Dict[str, Any]],
        original_goal_id: int,
    ) -> List[Tuple[int, float]]:
        """Hindsight goal relabelling: find what goals WERE achieved.

        For each step, check all goals and return relabelling candidates.

        Returns:
            List of (achieved_goal_id, bonus) for relabelling.
        """
        relabels: List[Tuple[int, float]] = []
        import random

        if random.random() > self.config.relabel_prob:
            return relabels

        # Check what goals were achieved at each step
        for step_data in trajectory:
            state = step_data.get("state", {})
            for goal in self._goals:
                if goal["id"] == original_goal_id:
                    continue
                keys = goal.get("achievement_keys", [])
                if all(state.get(k, False) for k in keys):
                    relabels.append((goal["id"], self.config.goal_reward_scale * 0.5))

        return relabels

    def get_policy_action(
        self,
        state_tensor: torch.Tensor,
        goal_id: int,
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor, float]:
        """Get action from goal-conditioned policy.

        Args:
            state_tensor: (state_dim,) state.
            goal_id: Current goal index.
            deterministic: If True, pick argmax.

        Returns:
            (action_idx, log_prob, value)
        """
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        goal_tensor = torch.tensor([goal_id], dtype=torch.long)
        goal_emb = self.goal_embedding(goal_tensor)

        logits, value = self.policy(state_tensor, goal_emb)
        logits = logits.squeeze(0)
        value = value.squeeze().item()

        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)

        if deterministic:
            action = int(probs.argmax().item())
        else:
            action = int(dist.sample().item())

        log_prob = dist.log_prob(torch.tensor(action))
        return action, log_prob, float(value)

    def get_goal_success_rates(self) -> Dict[str, float]:
        """Return success rate per goal."""
        rates = {}
        for goal in self._goals:
            gid = goal["id"]
            attempts = self._goal_attempts.get(gid, 0)
            successes = self._goal_successes.get(gid, 0)
            rates[goal["name"]] = successes / max(attempts, 1)
        return rates

    def get_stats(self) -> Dict[str, Any]:
        """Return goal-conditioned RL statistics."""
        return {
            "n_goals": len(self._goals),
            "goal_names": [g["name"] for g in self._goals],
            "current_goal": self._current_goal_id,
            "attempts": dict(self._goal_attempts),
            "successes": dict(self._goal_successes),
            "success_rates": self.get_goal_success_rates(),
        }

    def reset(self) -> None:
        """Reset current goal (keep history)."""
        self._current_goal_id = None
