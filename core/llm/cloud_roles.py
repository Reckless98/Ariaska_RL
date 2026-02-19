#!/usr/bin/env python3
"""
core/llm/cloud_roles.py — Phase 9.7: Cloud LLM acceleration roles

Defines specialized LLM roles for training acceleration:
- StrategicPlanner: Episode-level plan generation (replaces EC if enabled)
- TacticalAdvisor: Per-step command ranking/assessment
- JudgeRanker: Compare candidate commands and rank by expected value
- PostmortemSkillExtractor: Deep analysis of episode transcripts
- DAggerCorrector: Expert demonstrations for PPO correction

All roles are OFF by default (feature-flagged) and require explicit opt-in.
They consume LLM tokens so should only be used when budget allows.

Usage:
    from core.llm.cloud_roles import get_role, LLMRole
    planner = get_role(LLMRole.STRATEGIC_PLANNER, gpt_manager)
    if planner and planner.enabled:
        plan = planner.generate_plan(state, target_info)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.llm.cloud_roles")


class LLMRole(Enum):
    """Available cloud LLM acceleration roles."""
    STRATEGIC_PLANNER = "strategic_planner"
    TACTICAL_ADVISOR = "tactical_advisor"
    JUDGE_RANKER = "judge_ranker"
    POSTMORTEM_SKILLS = "postmortem_skills"
    DAGGER_CORRECTOR = "dagger_corrector"


@dataclass
class RoleConfig:
    """Configuration for an LLM role."""
    model: str = "gpt-5-mini"
    fallback_model: str = "gpt-5.2-mini"
    max_calls_per_episode: int = 5
    temperature: float = 0.3
    max_tokens: int = 500


class BaseLLMRole:
    """Base class for all LLM acceleration roles.

    All subclasses check their feature flag before executing.
    """

    def __init__(
        self,
        role: LLMRole,
        flag_name: str,
        gpt_manager: Any,
        config: Optional[RoleConfig] = None,
    ):
        self.role = role
        self.flag_name = flag_name
        self.gpt_manager = gpt_manager
        self.config = config or RoleConfig()
        self._calls_this_episode = 0
        self._total_calls = 0

    @property
    def enabled(self) -> bool:
        """Check feature flag."""
        try:
            from core.feature_flags import get_feature_flags
            return getattr(get_feature_flags(), self.flag_name, False)
        except Exception:
            return False

    def can_call(self) -> bool:
        """Check if budget allows another call this episode."""
        return (
            self.enabled
            and self._calls_this_episode < self.config.max_calls_per_episode
            and self.gpt_manager is not None
        )

    def reset_episode(self) -> None:
        self._calls_this_episode = 0

    def _call_llm(self, prompt: str, task_type: str = "tactical") -> str:
        """Make an LLM call through GPTManager."""
        if not self.can_call():
            return ""
        try:
            result = self.gpt_manager.gpt_request(
                prompt=prompt,
                task_type=task_type,
                agent_id=f"llm_role_{self.role.value}",
            )
            self._calls_this_episode += 1
            self._total_calls += 1
            return result or ""
        except Exception as e:
            logger.debug(f"[LLM-ROLE] {self.role.value} call failed: {e}")
            return ""

    def get_stats(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "enabled": self.enabled,
            "calls_this_episode": self._calls_this_episode,
            "total_calls": self._total_calls,
            "budget": self.config.max_calls_per_episode,
        }


class StrategicPlanner(BaseLLMRole):
    """Episode-level strategic plan generation.

    Generates a high-level attack plan at episode start based on
    target profile and campaign history. FF: llm_strategic_planner (OFF)
    """

    def __init__(self, gpt_manager: Any, config: Optional[RoleConfig] = None):
        super().__init__(
            role=LLMRole.STRATEGIC_PLANNER,
            flag_name="llm_strategic_planner",
            gpt_manager=gpt_manager,
            config=config or RoleConfig(max_calls_per_episode=2, temperature=0.4, max_tokens=800),
        )

    def generate_plan(
        self,
        target_ip: str,
        target_type: str,
        known_services: List[str],
        campaign_history: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Generate a strategic attack plan for the episode."""
        if not self.can_call():
            return {"plan": [], "reasoning": "LLM planner disabled or budget exhausted"}

        prompt = (
            f"As a strategic penetration testing planner, create an attack plan for:\n"
            f"Target: {target_ip} ({target_type})\n"
            f"Known services: {', '.join(known_services[:10])}\n"
            f"Prior campaigns: {len((campaign_history or {}).get('episodes', []))} episodes\n\n"
            f"Output a JSON object with 'phases' (ordered attack phases) and "
            f"'priority_services' (services to target first).\n"
            f"Focus on high-probability exploitation paths."
        )
        response = self._call_llm(prompt, task_type="strategic")
        return {"plan": response, "reasoning": "LLM strategic plan", "source": "llm_planner"}


class TacticalAdvisor(BaseLLMRole):
    """Per-step tactical command assessment.

    Evaluates candidate commands and provides tactical guidance.
    FF: llm_tactical_advisor (OFF)
    """

    def __init__(self, gpt_manager: Any, config: Optional[RoleConfig] = None):
        super().__init__(
            role=LLMRole.TACTICAL_ADVISOR,
            flag_name="llm_tactical_advisor",
            gpt_manager=gpt_manager,
            config=config or RoleConfig(max_calls_per_episode=10, temperature=0.2, max_tokens=300),
        )

    def assess_command(
        self,
        command: str,
        phase: str,
        discovery_board: Dict[str, Any],
        detection_risk: float = 0.0,
    ) -> Dict[str, Any]:
        """Assess a candidate command's tactical value."""
        if not self.can_call():
            return {"approved": True, "confidence": 0.5, "reasoning": "advisor disabled"}

        prompt = (
            f"Assess this penetration testing command:\n"
            f"Command: {command[:200]}\n"
            f"Phase: {phase}\n"
            f"Detection risk: {detection_risk:.1f}\n"
            f"Known ports: {len(discovery_board.get('ports', []))}\n\n"
            f"Is this command appropriate? Reply: APPROVE or REDIRECT with reasoning."
        )
        response = self._call_llm(prompt, task_type="tactical")
        approved = "APPROVE" in response.upper() if response else True
        return {
            "approved": approved,
            "confidence": 0.7 if approved else 0.3,
            "reasoning": response[:200] if response else "",
        }


class JudgeRanker(BaseLLMRole):
    """Compare and rank candidate commands by expected value.

    FF: llm_judge_ranker (OFF)
    """

    def __init__(self, gpt_manager: Any, config: Optional[RoleConfig] = None):
        super().__init__(
            role=LLMRole.JUDGE_RANKER,
            flag_name="llm_judge_ranker",
            gpt_manager=gpt_manager,
            config=config or RoleConfig(max_calls_per_episode=5, temperature=0.1, max_tokens=400),
        )

    def rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        phase: str,
        context: str = "",
    ) -> List[Dict[str, Any]]:
        """Rank candidate commands by expected value.

        Args:
            candidates: List of {command, template_name, source, confidence}
            phase: Current attack phase
            context: Additional context string

        Returns:
            Ranked list with added 'rank' and 'llm_score' fields
        """
        if not self.can_call() or not candidates:
            return candidates

        cand_text = "\n".join(
            f"{i+1}. [{c.get('source', '?')}] {c.get('template_name', '?')}: {c.get('command', '?')[:80]}"
            for i, c in enumerate(candidates[:5])
        )
        prompt = (
            f"Rank these penetration testing commands by expected value in {phase} phase:\n"
            f"{cand_text}\n\n"
            f"Context: {context[:200]}\n"
            f"Reply with the numbers in order of preference (best first)."
        )
        response = self._call_llm(prompt, task_type="tactical")
        # Parse ranking from response — fallback to original order
        for i, c in enumerate(candidates):
            c["llm_rank"] = i + 1
        return candidates


class PostmortemSkillExtractor(BaseLLMRole):
    """Extract skill cards from episode transcripts.

    FF: llm_postmortem_skills (OFF)
    """

    def __init__(self, gpt_manager: Any, config: Optional[RoleConfig] = None):
        super().__init__(
            role=LLMRole.POSTMORTEM_SKILLS,
            flag_name="llm_postmortem_skills",
            gpt_manager=gpt_manager,
            config=config or RoleConfig(
                model="gpt-5.2", max_calls_per_episode=1, temperature=0.3, max_tokens=1000
            ),
        )

    def extract_skills(
        self,
        transcript: str,
        total_reward: float,
        highest_phase: str,
    ) -> List[Dict[str, Any]]:
        """Extract reusable skill cards from episode transcript."""
        if not self.can_call():
            return []

        prompt = (
            f"Analyze this penetration testing episode transcript and extract reusable skills:\n"
            f"Reward: {total_reward:+.1f}, Highest phase: {highest_phase}\n\n"
            f"Transcript (last 20 actions):\n{transcript[:2000]}\n\n"
            f"For each skill, provide: trigger (phase+conditions), template (command), "
            f"expected_reward. Output as JSON array of skill objects."
        )
        response = self._call_llm(prompt, task_type="reasoning")
        return [{"raw_response": response, "source": "llm_postmortem"}]


class DAggerCorrector(BaseLLMRole):
    """Expert demonstrations for PPO correction via DAgger.

    When PPO repeatedly selects poor actions, the DAgger corrector
    provides expert demonstrations that are mixed into PPO replay buffer.

    FF: dagger_corrections (OFF)
    """

    def __init__(self, gpt_manager: Any, config: Optional[RoleConfig] = None):
        super().__init__(
            role=LLMRole.DAGGER_CORRECTOR,
            flag_name="dagger_corrections",
            gpt_manager=gpt_manager,
            config=config or RoleConfig(max_calls_per_episode=3, temperature=0.1, max_tokens=300),
        )

    def get_correction(
        self,
        state_description: str,
        ppo_action: str,
        ppo_reward: float,
        phase: str,
        available_commands: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Get expert correction for a poor PPO decision.

        Only called when PPO action received negative reward.
        """
        if not self.can_call() or ppo_reward >= 0:
            return None

        cmds = "\n".join(f"- {c}" for c in available_commands[:10])
        prompt = (
            f"The RL agent chose poorly in {phase} phase:\n"
            f"State: {state_description[:300]}\n"
            f"PPO chose: {ppo_action[:100]} (reward: {ppo_reward:.1f})\n\n"
            f"Available commands:\n{cmds}\n\n"
            f"Which command would an expert choose? Reply with just the command."
        )
        response = self._call_llm(prompt, task_type="tactical")
        if response:
            return {
                "expert_command": response.strip()[:200],
                "source": "dagger",
                "original_action": ppo_action[:100],
                "original_reward": ppo_reward,
            }
        return None


# ── Factory function ────────────────────────────────────────────────────────

_role_cache: Dict[str, BaseLLMRole] = {}


def get_role(role: LLMRole, gpt_manager: Any) -> Optional[BaseLLMRole]:
    """Get or create an LLM role instance.

    Returns None if the role's feature flag is OFF.
    """
    key = role.value
    if key not in _role_cache:
        cls_map = {
            LLMRole.STRATEGIC_PLANNER: StrategicPlanner,
            LLMRole.TACTICAL_ADVISOR: TacticalAdvisor,
            LLMRole.JUDGE_RANKER: JudgeRanker,
            LLMRole.POSTMORTEM_SKILLS: PostmortemSkillExtractor,
            LLMRole.DAGGER_CORRECTOR: DAggerCorrector,
        }
        cls = cls_map.get(role)
        if cls:
            _role_cache[key] = cls(gpt_manager=gpt_manager)
    
    instance = _role_cache.get(key)
    if instance and instance.enabled:
        return instance
    return None


def reset_role_cache() -> None:
    """Clear the role instance cache (for testing)."""
    _role_cache.clear()
