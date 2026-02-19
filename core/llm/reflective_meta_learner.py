"""core/llm/reflective_meta_learner.py — Phase 41: Reflective Meta-Learning.

After each episode, asks GPTManager to reflect on performance and
generates strategy adjustments that are injected as context into
subsequent LLM calls.
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger("ariaska.llm.reflective_meta_learner")


@dataclass
class ReflectionConfig:
    """Configuration for reflective meta-learning."""
    enabled: bool = True
    reflect_every_n_episodes: int = 1
    max_reflection_tokens: int = 500
    history_window: int = 10


@dataclass
class ReflectionResult:
    """Output of a reflection cycle."""
    insights: List[str] = field(default_factory=list)
    strategy_adjustments: List[str] = field(default_factory=list)
    phase_specific_tips: Dict[str, str] = field(default_factory=dict)
    avoid_commands: List[str] = field(default_factory=list)
    promising_chains: List[str] = field(default_factory=list)
    episode: int = 0
    timestamp: float = 0.0


class ReflectiveMetaLearner:
    """Episode-level reflection engine.

    After an episode completes, ``reflect_on_episode`` calls GPTManager
    to analyze what went well/badly and produces structured insights.
    ``get_context_injection`` formats recent insights for LLM system prompts.
    """

    def __init__(self, config: Optional[ReflectionConfig] = None) -> None:
        self.config = config or ReflectionConfig()
        self._history: Deque[ReflectionResult] = deque(
            maxlen=self.config.history_window
        )
        self._episode_counter: int = 0

    def reflect_on_episode(
        self,
        episode_data: Dict[str, Any],
        gpt_manager: Optional[Any] = None,
    ) -> ReflectionResult:
        """Reflect on a completed episode.

        Args:
            episode_data: Episode metrics (reward, steps, phase, discoveries, etc.)
            gpt_manager: Optional GPTManager for LLM reflection.

        Returns:
            Structured reflection result.
        """
        self._episode_counter += 1
        result = ReflectionResult(
            episode=self._episode_counter,
            timestamp=time.time(),
        )

        if not self.config.enabled:
            return result

        if self._episode_counter % self.config.reflect_every_n_episodes != 0:
            return result

        # Build offline reflection (always works, no LLM needed)
        reward = episode_data.get("total_reward", 0.0)
        steps = episode_data.get("steps", 0)
        phase = episode_data.get("highest_phase", "RECON")
        discoveries = episode_data.get("total_discoveries", 0)

        if reward < 0:
            result.insights.append(
                f"Negative reward ({reward:.1f}) — too many failed commands"
            )
            result.strategy_adjustments.append(
                "Reduce exploration, focus on known-good command patterns"
            )
        if steps > 40 and phase in ("RECON", "ENUMERATION"):
            result.insights.append(
                f"Stuck in {phase} after {steps} steps — phase transition blocked"
            )
            result.strategy_adjustments.append(
                "Lower phase transition threshold or force-advance"
            )
        if discoveries == 0:
            result.insights.append("Zero discoveries — scanning may be misconfigured")
            result.avoid_commands.append("Avoid repeating the same scan type")

        # LLM reflection (if available and budget allows)
        if gpt_manager is not None:
            try:
                prompt = self._build_reflection_prompt(episode_data)
                response = gpt_manager.gpt_request(
                    prompt,
                    task_type="learning",
                    agent_id="meta_learner",
                    max_tokens=self.config.max_reflection_tokens,
                )
                if response:
                    self._parse_llm_reflection(response, result)
            except Exception as exc:
                logger.debug("LLM reflection failed: %s", exc)

        self._history.append(result)
        logger.debug(
            "Reflection ep=%d: %d insights, %d adjustments",
            self._episode_counter, len(result.insights),
            len(result.strategy_adjustments),
        )
        return result

    def get_context_injection(self, last_n: int = 3) -> str:
        """Format recent reflection insights for LLM system prompt injection.

        Args:
            last_n: Number of recent reflections to include.

        Returns:
            Formatted string for system prompt context.
        """
        if not self._history:
            return ""

        entries = list(self._history)[-last_n:]
        lines: List[str] = ["[REFLECTION CONTEXT]"]
        for r in entries:
            if r.insights:
                lines.append(f"Episode {r.episode} insights:")
                for ins in r.insights[:3]:
                    lines.append(f"  - {ins}")
            if r.strategy_adjustments:
                lines.append("Strategy adjustments:")
                for adj in r.strategy_adjustments[:2]:
                    lines.append(f"  - {adj}")
            if r.avoid_commands:
                lines.append(f"Avoid: {', '.join(r.avoid_commands[:3])}")
        return "\n".join(lines)

    @property
    def history(self) -> List[ReflectionResult]:
        """Return copy of reflection history."""
        return list(self._history)

    def _build_reflection_prompt(self, data: Dict[str, Any]) -> str:
        """Build the reflection prompt for GPTManager."""
        return (
            f"Reflect on this penetration testing episode:\n"
            f"Phase reached: {data.get('highest_phase', 'RECON')}\n"
            f"Steps taken: {data.get('steps', 0)}\n"
            f"Total reward: {data.get('total_reward', 0.0):.1f}\n"
            f"Discoveries: {data.get('total_discoveries', 0)}\n"
            f"Stagnation steps: {data.get('stagnation_steps', 0)}\n"
            f"\nProvide: 1) Key insights, 2) Strategy adjustments, "
            f"3) Commands to avoid, 4) Promising attack chains.\n"
            f"Respond in JSON with keys: insights, strategy_adjustments, "
            f"avoid_commands, promising_chains."
        )

    def _parse_llm_reflection(
        self, response: str, result: ReflectionResult
    ) -> None:
        """Parse LLM reflection response defensively."""
        try:
            data = json.loads(response)
            if isinstance(data.get("insights"), list):
                result.insights.extend(str(i) for i in data["insights"][:5])
            if isinstance(data.get("strategy_adjustments"), list):
                result.strategy_adjustments.extend(
                    str(a) for a in data["strategy_adjustments"][:3]
                )
            if isinstance(data.get("avoid_commands"), list):
                result.avoid_commands.extend(
                    str(c) for c in data["avoid_commands"][:5]
                )
            if isinstance(data.get("promising_chains"), list):
                result.promising_chains.extend(
                    str(c) for c in data["promising_chains"][:5]
                )
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.debug("Could not parse LLM reflection JSON: %s", exc)
            # Fall back: treat as free-text insight
            if response and len(response) < 2000:
                result.insights.append(response[:500])
