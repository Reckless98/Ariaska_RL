"""
Reflective Cortex — Batch LLM Meta-Learning System for Ariaska_RL.

Phase 9.2: Analyzes episode history every N episodes and generates
meta-insights that improve agent decision-making, mentor prompts,
and knowledge graph enrichment.

Architecture:
  - ReflectiveCortex: Main orchestrator, runs every `reflect_interval` episodes
  - ReflectionResult: Structured output from each reflection cycle
  - StrategicInsight: Individual insight with confidence and applicability

Workflow:
  1. Collect episode transcripts from last N episodes (JSONL logs)
  2. Aggregate metrics: phase progression, discovery rates, command diversity
  3. Call GPT with structured analysis prompt (budget-gated)
  4. Parse response into StrategicInsight objects
  5. Inject insights into KG as methodology nodes
  6. Update SmartMentor system prompt with top insights
  7. Adjust SmartCoach parameters based on identified patterns

Usage:
    from core.llm.reflective_cortex import ReflectiveCortex
    cortex = ReflectiveCortex(gpt_manager=gpt, knowledge_graph=kg)
    result = cortex.reflect(episode_history=last_10_episodes)
    cortex.apply_insights(result, smart_coach=coach, mentor=mentor)
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.gpt_manager import GPTManager
    from core.knowledge.kg_manager import KnowledgeGraph

logger = logging.getLogger("ariaska.reflective_cortex")


# ─── Data Structures ────────────────────────────────────────────────────────

@dataclass
class StrategicInsight:
    """A single insight from reflective analysis."""
    insight_id: str
    category: str                # "strategy", "exploit_path", "defense_gap", "efficiency", "knowledge_gap"
    description: str
    confidence: float = 0.5      # 0.0-1.0
    applicability: str = "all"   # "all", "ms2", "ms3", "htb", "generic"
    recommended_actions: List[str] = field(default_factory=list)
    affected_agents: List[str] = field(default_factory=list)  # "red", "scout", "shadow", etc.
    source_episodes: List[int] = field(default_factory=list)
    priority: int = 5            # 1-10, higher = more urgent
    created_at: float = field(default_factory=time.time)

    def to_prompt_fragment(self) -> str:
        """Convert to a prompt-injectable string."""
        actions = "; ".join(self.recommended_actions[:3]) if self.recommended_actions else ""
        return (
            f"[{self.category.upper()} | conf={self.confidence:.0%} | p={self.priority}] "
            f"{self.description}"
            + (f" → Actions: {actions}" if actions else "")
        )


@dataclass
class ReflectionResult:
    """Complete result from a reflection cycle."""
    cycle_id: int
    episode_range: tuple = (0, 0)  # (start_ep, end_ep)
    insights: List[StrategicInsight] = field(default_factory=list)
    aggregate_metrics: Dict[str, Any] = field(default_factory=dict)
    parameter_adjustments: Dict[str, Any] = field(default_factory=dict)
    mentor_prompt_additions: List[str] = field(default_factory=list)
    knowledge_gaps: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    tokens_used: int = 0

    @property
    def top_insights(self) -> List[StrategicInsight]:
        """Get insights sorted by priority × confidence."""
        return sorted(
            self.insights,
            key=lambda i: i.priority * i.confidence,
            reverse=True,
        )[:5]


# ─── Reflective Cortex ──────────────────────────────────────────────────────

class ReflectiveCortex:
    """
    Batch meta-learning system that periodically analyzes training history
    and generates strategic insights to improve agent behavior.

    Runs every `reflect_interval` episodes and produces:
    - Strategic insights for mentor injection
    - Knowledge graph enrichment
    - SmartCoach parameter adjustments
    - Identified knowledge gaps for targeted scraping
    """

    def __init__(
        self,
        gpt_manager: Optional[GPTManager] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        reflect_interval: int = 10,
        max_history_episodes: int = 20,
        output_dir: Optional[str] = None,
        enable_llm: bool = True,
    ):
        self.gpt_manager = gpt_manager
        self.knowledge_graph = knowledge_graph
        self.reflect_interval = reflect_interval
        self.max_history_episodes = max_history_episodes
        self.enable_llm = enable_llm
        self.output_dir = Path(output_dir) if output_dir else (
            Path(__file__).parent.parent.parent / "postmortems" / "reflections"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # State
        self._cycle_count = 0
        self._all_insights: List[StrategicInsight] = []
        self._last_reflection_episode = 0
        self._accumulated_episodes: List[Dict[str, Any]] = []

        # Reflection prompt template
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        return """You are Ariaska's Reflective Cortex — a meta-learning analyst for an autonomous 
multi-agent cybersecurity RL system. You analyze batches of training episodes and produce 
structured strategic insights.

Your analysis should identify:
1. **Attack patterns that work** — which command sequences lead to phase progression
2. **Repeated failures** — commands that consistently fail and should be deprioritized
3. **Knowledge gaps** — situations where agents lack knowledge of effective techniques
4. **Efficiency improvements** — faster paths to exploitation from current discoveries
5. **Agent coordination issues** — where agent handoffs fail or overlap

Output format (strict JSON array):
[
  {
    "category": "strategy|exploit_path|defense_gap|efficiency|knowledge_gap",
    "description": "Clear, actionable insight",
    "confidence": 0.0-1.0,
    "recommended_actions": ["action1", "action2"],
    "affected_agents": ["red", "scout", "shadow", "blue", "orion"],
    "priority": 1-10
  }
]

Be specific and actionable. Reference actual commands, services, and ports when possible.
Focus on insights that can directly improve the next batch of training episodes."""

    # ─── Core Reflection ─────────────────────────────────────────────────

    def should_reflect(self, episode_number: int) -> bool:
        """Check if it's time for a reflection cycle."""
        episodes_since = episode_number - self._last_reflection_episode
        return episodes_since >= self.reflect_interval

    def accumulate_episode(self, episode_data: Dict[str, Any]):
        """Add episode data to accumulation buffer for next reflection."""
        self._accumulated_episodes.append(episode_data)
        # Keep only recent history
        if len(self._accumulated_episodes) > self.max_history_episodes:
            self._accumulated_episodes = self._accumulated_episodes[-self.max_history_episodes:]

    def reflect(
        self,
        episode_history: Optional[List[Dict[str, Any]]] = None,
        current_episode: int = 0,
    ) -> ReflectionResult:
        """
        Run a reflection cycle over recent episode history.

        Args:
            episode_history: List of episode data dicts (from SmartOrchestrator)
            current_episode: Current episode number for tracking

        Returns:
            ReflectionResult with insights and parameter adjustments
        """
        start = time.time()
        self._cycle_count += 1
        self._last_reflection_episode = current_episode

        history = episode_history or self._accumulated_episodes
        if not history:
            logger.info("No episode history to reflect on")
            return ReflectionResult(cycle_id=self._cycle_count)

        logger.info(
            f"Reflective Cortex cycle #{self._cycle_count}: "
            f"analyzing {len(history)} episodes"
        )

        # 1. Aggregate metrics from history
        metrics = self._aggregate_metrics(history)

        # 2. Generate insights (LLM or heuristic fallback)
        if self.enable_llm and self.gpt_manager:
            insights = self._llm_reflection(history, metrics)
        else:
            insights = self._heuristic_reflection(history, metrics)

        # 3. Compute parameter adjustments from insights
        param_adjustments = self._compute_adjustments(insights, metrics)

        # 4. Extract mentor prompt additions
        mentor_additions = [
            i.to_prompt_fragment()
            for i in sorted(insights, key=lambda x: x.priority * x.confidence, reverse=True)[:3]
        ]

        # 5. Identify knowledge gaps
        knowledge_gaps = [
            i.description for i in insights
            if i.category == "knowledge_gap"
        ]

        result = ReflectionResult(
            cycle_id=self._cycle_count,
            episode_range=(
                min(ep.get("episode_number", 0) for ep in history),
                max(ep.get("episode_number", 0) for ep in history),
            ),
            insights=insights,
            aggregate_metrics=metrics,
            parameter_adjustments=param_adjustments,
            mentor_prompt_additions=mentor_additions,
            knowledge_gaps=knowledge_gaps,
            duration_seconds=time.time() - start,
        )

        # Store insights for future reference
        self._all_insights.extend(insights)

        # Save reflection result
        self._save_reflection(result)

        logger.info(
            f"Reflection cycle #{self._cycle_count} complete: "
            f"{len(insights)} insights, {len(param_adjustments)} adjustments, "
            f"{result.duration_seconds:.1f}s"
        )

        return result

    # ─── Metric Aggregation ──────────────────────────────────────────────

    def _aggregate_metrics(self, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across episode history."""
        metrics = {
            "total_episodes": len(episodes),
            "avg_reward": 0.0,
            "avg_steps": 0.0,
            "phase_distribution": Counter(),
            "command_frequency": Counter(),
            "discovery_types": Counter(),
            "agent_decision_sources": Counter(),
            "failure_commands": Counter(),
            "success_commands": Counter(),
            "avg_diversity_ratio": 0.0,
            "exfiltration_rate": 0.0,
            "avg_unique_commands": 0.0,
        }

        total_reward = 0.0
        total_steps = 0
        total_diversity = 0.0
        exfiltration_count = 0
        total_unique = 0

        for ep in episodes:
            ep_reward = ep.get("total_reward", ep.get("reward", 0))
            ep_steps = ep.get("steps", ep.get("total_steps", 0))
            total_reward += ep_reward
            total_steps += ep_steps

            # Phase reached
            highest_phase = ep.get("highest_phase", ep.get("phase", "RECON"))
            metrics["phase_distribution"][highest_phase] += 1
            if highest_phase == "EXFILTRATION":
                exfiltration_count += 1

            # Command frequency
            commands = ep.get("commands_used", ep.get("commands", []))
            if isinstance(commands, list):
                for cmd in commands:
                    if isinstance(cmd, str):
                        tool = cmd.split()[0] if cmd.split() else cmd
                        metrics["command_frequency"][tool] += 1
                    elif isinstance(cmd, dict):
                        tool = cmd.get("tool", cmd.get("command", "")).split()[0]
                        metrics["command_frequency"][tool] += 1

            # Decision sources
            sources = ep.get("decision_sources", {})
            if isinstance(sources, dict):
                for src, count in sources.items():
                    metrics["agent_decision_sources"][src] += count

            # Discoveries
            discoveries = ep.get("discoveries", {})
            if isinstance(discoveries, dict):
                for dtype, count in discoveries.items():
                    metrics["discovery_types"][dtype] += (count if isinstance(count, int) else len(count))

            # Diversity
            diversity = ep.get("diversity_ratio", 0)
            total_diversity += diversity
            unique = ep.get("unique_commands", 0)
            total_unique += unique

        n = max(len(episodes), 1)
        metrics["avg_reward"] = total_reward / n
        metrics["avg_steps"] = total_steps / n
        metrics["avg_diversity_ratio"] = total_diversity / n
        metrics["exfiltration_rate"] = exfiltration_count / n
        metrics["avg_unique_commands"] = total_unique / n

        # Convert Counters to dicts for JSON serialization
        metrics["phase_distribution"] = dict(metrics["phase_distribution"])
        metrics["command_frequency"] = dict(metrics["command_frequency"].most_common(30))
        metrics["discovery_types"] = dict(metrics["discovery_types"])
        metrics["agent_decision_sources"] = dict(metrics["agent_decision_sources"])
        metrics["failure_commands"] = dict(metrics["failure_commands"].most_common(20))
        metrics["success_commands"] = dict(metrics["success_commands"].most_common(20))

        return metrics

    # ─── LLM Reflection ─────────────────────────────────────────────────

    def _llm_reflection(
        self,
        episodes: List[Dict[str, Any]],
        metrics: Dict[str, Any],
    ) -> List[StrategicInsight]:
        """Use GPT to analyze episode history and generate insights."""
        if not self.gpt_manager:
            return self._heuristic_reflection(episodes, metrics)

        # Build analysis prompt
        prompt = self._build_analysis_prompt(episodes, metrics)

        try:
            # Check budget
            if hasattr(self.gpt_manager, 'can_make_request'):
                if not self.gpt_manager.can_make_request():
                    logger.info("GPT budget exceeded — falling back to heuristic reflection")
                    return self._heuristic_reflection(episodes, metrics)

            response = self.gpt_manager.gpt_request(
                prompt=prompt,
                task_type="reasoning",
                agent_id="ReflectiveCortex",
                system_prompt=self._system_prompt,
            )

            if not response:
                return self._heuristic_reflection(episodes, metrics)

            # Parse JSON response
            insights = self._parse_llm_response(response, episodes)
            logger.info(f"LLM reflection produced {len(insights)} insights")
            return insights

        except Exception as e:
            logger.warning(f"LLM reflection failed: {e}")
            return self._heuristic_reflection(episodes, metrics)

    def _build_analysis_prompt(
        self,
        episodes: List[Dict[str, Any]],
        metrics: Dict[str, Any],
    ) -> str:
        """Build structured prompt for LLM analysis."""
        # Compact episode summaries
        ep_summaries = []
        for ep in episodes[-10:]:  # Last 10 episodes max
            summary = {
                "ep": ep.get("episode_number", "?"),
                "reward": round(ep.get("total_reward", 0), 1),
                "steps": ep.get("steps", ep.get("total_steps", 0)),
                "phase": ep.get("highest_phase", "?"),
                "unique_cmds": ep.get("unique_commands", 0),
                "discoveries": ep.get("total_discoveries", 0),
            }
            # Add top commands (compact)
            commands = ep.get("commands_used", [])
            if isinstance(commands, list):
                summary["cmds"] = [
                    (c.split()[0] if isinstance(c, str) and c.split() else str(c))
                    for c in commands[:15]
                ]
            ep_summaries.append(summary)

        prompt = f"""Analyze these {len(episodes)} training episodes from an autonomous pentesting RL system.

## Aggregate Metrics
{json.dumps(metrics, indent=1, default=str)}

## Episode Summaries (last {len(ep_summaries)})
{json.dumps(ep_summaries, indent=1, default=str)}

## Key Questions
1. What command sequences consistently lead to phase advancement?
2. Which commands are used frequently but never produce discoveries?
3. Are there obvious exploit paths being missed? (services found but never exploited)
4. How can agent coordination be improved?
5. What knowledge is the system missing?

Provide 3-8 structured insights as a JSON array."""

        return prompt

    def _parse_llm_response(
        self,
        response: str,
        episodes: List[Dict[str, Any]],
    ) -> List[StrategicInsight]:
        """Parse LLM response into StrategicInsight objects."""
        insights = []

        # Try to extract JSON array from response
        import re
        json_match = re.search(r'\[[\s\S]*\]', response)
        if not json_match:
            logger.debug("No JSON array found in LLM response")
            return insights

        try:
            items = json.loads(json_match.group())
            for i, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                insight = StrategicInsight(
                    insight_id=f"llm-{self._cycle_count}-{i}",
                    category=item.get("category", "strategy"),
                    description=item.get("description", ""),
                    confidence=min(max(float(item.get("confidence", 0.5)), 0.0), 1.0),
                    recommended_actions=item.get("recommended_actions", []),
                    affected_agents=item.get("affected_agents", []),
                    priority=min(max(int(item.get("priority", 5)), 1), 10),
                    source_episodes=[ep.get("episode_number", 0) for ep in episodes[-5:]],
                )
                insights.append(insight)
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"JSON parse error: {e}")

        return insights

    # ─── Heuristic Reflection (Offline Fallback) ─────────────────────────

    def _heuristic_reflection(
        self,
        episodes: List[Dict[str, Any]],
        metrics: Dict[str, Any],
    ) -> List[StrategicInsight]:
        """Generate insights without LLM using pattern analysis."""
        insights = []

        # 1. Low diversity detection
        avg_diversity = metrics.get("avg_diversity_ratio", 0)
        if avg_diversity < 0.3:
            insights.append(StrategicInsight(
                insight_id=f"heur-{self._cycle_count}-diversity",
                category="efficiency",
                description=f"Command diversity is low ({avg_diversity:.0%}). "
                            f"Agents are repeating too many commands. "
                            f"Consider increasing anti-repeat penalty or expanding command pool.",
                confidence=0.8,
                recommended_actions=[
                    "Increase redundancy_penalty in SmartRewardCalculator",
                    "Add more alternative commands to anti-repeat pool",
                    "Boost PPO entropy coefficient temporarily",
                ],
                affected_agents=["red", "scout"],
                priority=7,
            ))

        # 2. Stalled phase progression
        phase_dist = metrics.get("phase_distribution", {})
        exfil_rate = metrics.get("exfiltration_rate", 0)
        if exfil_rate < 0.5 and metrics.get("total_episodes", 0) > 5:
            # Check which phase is the bottleneck
            stuck_phase = max(phase_dist, key=phase_dist.get) if phase_dist else "RECON"
            insights.append(StrategicInsight(
                insight_id=f"heur-{self._cycle_count}-stuck",
                category="strategy",
                description=f"Exfiltration rate is only {exfil_rate:.0%}. "
                            f"Most episodes stall at {stuck_phase}. "
                            f"Agents need better {stuck_phase}→next phase transition strategies.",
                confidence=0.7,
                recommended_actions=[
                    f"Inject specific {stuck_phase.lower()} commands into mentor prompts",
                    f"Add playbook steps for transitioning from {stuck_phase}",
                    "Review SmartCoach stuck detection thresholds",
                ],
                affected_agents=["red", "orion"],
                priority=8,
            ))

        # 3. Overused commands
        cmd_freq = metrics.get("command_frequency", {})
        if cmd_freq:
            total_cmds = sum(cmd_freq.values())
            for cmd, count in list(cmd_freq.items())[:5]:
                if total_cmds > 0 and count / total_cmds > 0.15:
                    insights.append(StrategicInsight(
                        insight_id=f"heur-{self._cycle_count}-overuse-{cmd}",
                        category="efficiency",
                        description=f"Command '{cmd}' is used {count / total_cmds:.0%} of the time. "
                                    f"This suggests over-reliance. Consider deprioritizing or "
                                    f"adding negative reward for excessive use.",
                        confidence=0.6,
                        recommended_actions=[
                            f"Add '{cmd}' to redundancy watchlist",
                            "Increase novelty bonus for unexplored commands",
                        ],
                        priority=5,
                    ))

        # 4. Undiscovered services
        discovery_types = metrics.get("discovery_types", {})
        services_found = discovery_types.get("service", 0) + discovery_types.get("services", 0)
        credentials_found = discovery_types.get("credential", 0) + discovery_types.get("credentials", 0)
        if services_found > 5 and credentials_found == 0:
            insights.append(StrategicInsight(
                insight_id=f"heur-{self._cycle_count}-nocreds",
                category="knowledge_gap",
                description=f"Found {services_found} services but 0 credentials. "
                            f"Agents should attempt default credential testing "
                            f"and brute-force on discovered services.",
                confidence=0.75,
                recommended_actions=[
                    "Inject default credential lists for common services",
                    "Prioritize hydra/medusa for discovered SSH/FTP/MySQL",
                    "Add credential discovery to playbook chain",
                ],
                affected_agents=["red", "scout"],
                priority=7,
            ))

        # 5. Decision source imbalance
        sources = metrics.get("agent_decision_sources", {})
        anti_repeat_pct = sources.get("anti_repeat", 0)
        total_decisions = sum(sources.values()) if sources else 1
        if total_decisions > 0 and anti_repeat_pct / total_decisions > 0.4:
            insights.append(StrategicInsight(
                insight_id=f"heur-{self._cycle_count}-antirepeat",
                category="efficiency",
                description=f"Anti-repeat guard fires {anti_repeat_pct / total_decisions:.0%} of decisions. "
                            f"PPO is not learning from blocked actions. "
                            f"Fix: store anti-repeat actions with negative reward for PPO learning.",
                confidence=0.85,
                recommended_actions=[
                    "Store anti-repeat blocked actions in PPO trajectory with reward -5.0",
                    "Increase command pool diversity",
                    "Reduce PPO clip epsilon temporarily for more aggressive learning",
                ],
                affected_agents=["red"],
                priority=9,
            ))

        return insights

    # ─── Parameter Adjustments ───────────────────────────────────────────

    def _compute_adjustments(
        self,
        insights: List[StrategicInsight],
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute SmartCoach/PPO parameter adjustments from insights."""
        adjustments = {}

        for insight in insights:
            if insight.category == "efficiency" and insight.priority >= 7:
                if "diversity" in insight.description.lower():
                    adjustments["entropy_coef_boost"] = 0.005  # Temporary exploration boost
                    adjustments["novelty_weight"] = 2.0  # Increase novelty bonus
                if "anti-repeat" in insight.description.lower():
                    adjustments["store_antirepeat_negative"] = True

            if insight.category == "strategy" and "stall" in insight.description.lower():
                adjustments["stuck_threshold_reduce"] = 0.8  # Lower stuck detection

            if insight.category == "knowledge_gap":
                if "credential" in insight.description.lower():
                    adjustments["prioritize_credential_commands"] = True

        return adjustments

    # ─── Application ─────────────────────────────────────────────────────

    def apply_insights(
        self,
        result: ReflectionResult,
        smart_coach: Optional[Any] = None,
        mentor: Optional[Any] = None,
        knowledge_graph: Optional[Any] = None,
    ):
        """Apply reflection insights to system components."""
        kg = knowledge_graph or self.knowledge_graph

        # 1. Inject top insights into KG as methodology nodes
        if kg:
            for insight in result.top_insights:
                try:
                    kg.add_node(
                        f"insight:{insight.insight_id}",
                        node_type="methodology",
                        label=insight.description[:100],
                        category=insight.category,
                        description=insight.description,
                        priority=insight.priority,
                        confidence=insight.confidence,
                        actions=insight.recommended_actions,
                        source="reflective_cortex",
                    )
                except Exception as e:
                    logger.debug(f"Failed to add insight to KG: {e}")

        # 2. Add mentor prompt fragments
        if mentor and hasattr(mentor, 'add_reflective_insights'):
            mentor.add_reflective_insights(result.mentor_prompt_additions)
        elif mentor and hasattr(mentor, '_reflective_insights'):
            mentor._reflective_insights = result.mentor_prompt_additions

        # 3. Apply parameter adjustments to SmartCoach
        if smart_coach and result.parameter_adjustments:
            self._apply_coach_adjustments(smart_coach, result.parameter_adjustments)

        logger.info(
            f"Applied {len(result.top_insights)} insights, "
            f"{len(result.parameter_adjustments)} parameter adjustments"
        )

    def _apply_coach_adjustments(self, coach: Any, adjustments: Dict[str, Any]):
        """Apply parameter adjustments to SmartCoach."""
        try:
            if adjustments.get("entropy_coef_boost") and hasattr(coach, '_ppo'):
                ppo = coach._ppo
                if ppo and hasattr(ppo, 'config'):
                    old_entropy = ppo.config.entropy_coef
                    ppo.config.entropy_coef = min(old_entropy + adjustments["entropy_coef_boost"], 0.05)
                    logger.info(f"PPO entropy_coef: {old_entropy} → {ppo.config.entropy_coef}")

            if adjustments.get("novelty_weight") and hasattr(coach, '_reward_calculator'):
                rc = coach._reward_calculator
                if rc and hasattr(rc, 'novelty_weight'):
                    rc.novelty_weight = adjustments["novelty_weight"]

        except Exception as e:
            logger.debug(f"Coach adjustment failed: {e}")

    # ─── Persistence ─────────────────────────────────────────────────────

    def _save_reflection(self, result: ReflectionResult):
        """Save reflection result to disk."""
        try:
            filepath = self.output_dir / f"reflection_{result.cycle_id:04d}.json"
            data = {
                "cycle_id": result.cycle_id,
                "episode_range": result.episode_range,
                "insights": [asdict(i) for i in result.insights],
                "aggregate_metrics": result.aggregate_metrics,
                "parameter_adjustments": result.parameter_adjustments,
                "mentor_prompt_additions": result.mentor_prompt_additions,
                "knowledge_gaps": result.knowledge_gaps,
                "duration_seconds": result.duration_seconds,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            filepath.write_text(json.dumps(data, indent=2, default=str))
            logger.debug(f"Reflection saved: {filepath}")
        except Exception as e:
            logger.debug(f"Failed to save reflection: {e}")

    def get_active_insights(self, top_k: int = 5) -> List[StrategicInsight]:
        """Get the most relevant active insights for prompt injection."""
        if not self._all_insights:
            return []
        # Sort by recency × priority × confidence
        scored = []
        now = time.time()
        for insight in self._all_insights:
            age_hours = (now - insight.created_at) / 3600
            recency_factor = max(0.1, 1.0 - age_hours / 24)  # Decay over 24h
            score = insight.priority * insight.confidence * recency_factor
            scored.append((score, insight))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [insight for _, insight in scored[:top_k]]

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "cycles": self._cycle_count,
            "total_insights": len(self._all_insights),
            "active_insights": len(self.get_active_insights()),
            "last_reflection_episode": self._last_reflection_episode,
            "accumulated_episodes": len(self._accumulated_episodes),
        }

    def __repr__(self) -> str:
        return (
            f"<ReflectiveCortex cycles={self._cycle_count} "
            f"insights={len(self._all_insights)}>"
        )
