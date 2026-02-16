"""
Unified Cognitive Bus — Phase 9: Shared memory timeline for all algorithms and agents.

This is the central nervous system of Ariaska_RL. Every algorithm (PPO, SAC, DDQN, RND),
every agent (Red, Blue, Scout, Shadow, Orion), every LLM layer (Venice, Codex, Mentor),
and every knowledge system (SkillLibrary, Playbooks, CommandRegistry) reads and writes
to this unified timeline.

Architecture:
    CognitiveBus (singleton)
    ├── UnifiedTimeline (chronological event stream)
    │   ├── CognitiveEvent (timestamped, typed, attributed)
    │   └── EventType (action, discovery, reasoning, insight, directive, learning)
    ├── SharedBeliefState (current consensus reality)
    │   ├── target_model (what we know about the target)
    │   ├── attack_progress (where we are in the kill chain)
    │   └── confidence_map (per-service exploitation confidence)
    ├── ReasoningChain (WHY/WHEN/HOW behind decisions)
    │   ├── per-agent reasoning traces
    │   └── cross-agent reasoning fusion
    ├── VeniceInsightBridge (aha moments → mentor/codex prompts)
    │   ├── aha_memory (persistent cross-episode)
    │   └── discovery_correlations (pattern matching)
    └── LearningSignalAggregator (PPO/SAC/DDQN gradient feedback)
        ├── per-algorithm performance tracking
        └── Orion optimization suggestions

Author: Filip Volf
Phase: 9 — Unified Cognitive Architecture
"""

import logging
import time
import threading
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque
from pathlib import Path

logger = logging.getLogger("ariaska.cognitive_bus")


# ─────────────────────────── Event Types ───────────────────────────

class EventType(Enum):
    """Types of cognitive events on the unified timeline."""
    ACTION = auto()          # Agent executed a command
    DISCOVERY = auto()       # New information discovered
    REASONING = auto()       # WHY/WHEN/HOW reasoning trace
    INSIGHT = auto()         # Venice/Mentor aha moment or correlation
    DIRECTIVE = auto()       # Orion strategic directive
    LEARNING = auto()        # Algorithm learning signal (PPO update, etc.)
    PHASE_CHANGE = auto()    # Kill chain phase transition
    COMMUNICATION = auto()   # Inter-agent message
    ADAPTATION = auto()      # Hyperparameter or strategy adjustment
    FAILURE = auto()         # Failed attempt with analysis


@dataclass
class CognitiveEvent:
    """A single event on the unified cognitive timeline."""
    timestamp: float
    event_type: EventType
    agent_id: str
    episode: int
    step: int
    content: Dict[str, Any]
    reasoning: Optional[str] = None  # WHY this happened
    confidence: float = 0.5
    source_algorithm: Optional[str] = None  # ppo, sac, ddqn, mentor, venice, codex
    correlations: List[str] = field(default_factory=list)  # IDs of related events
    persistence: str = "episode"  # episode, session, permanent

    @property
    def event_id(self) -> str:
        return f"{self.agent_id}:{self.episode}:{self.step}:{self.event_type.name}"


# ─────────────────────────── Shared Belief State ───────────────────────────

@dataclass
class ServiceBelief:
    """What we believe about a specific service on the target."""
    port: int
    service: str
    version: Optional[str] = None
    exploitable: bool = False
    exploitation_confidence: float = 0.0
    known_cves: List[str] = field(default_factory=list)
    attempted_exploits: List[str] = field(default_factory=list)
    successful_exploits: List[str] = field(default_factory=list)
    credentials_found: List[str] = field(default_factory=list)
    shell_obtained: bool = False
    privesc_possible: bool = False
    notes: List[str] = field(default_factory=list)


@dataclass
class TargetModel:
    """Unified belief model of the target system."""
    ip: str = ""
    os_guess: str = "unknown"
    os_confidence: float = 0.0
    services: Dict[int, ServiceBelief] = field(default_factory=dict)
    open_ports: Set[int] = field(default_factory=set)
    discovered_users: Set[str] = field(default_factory=set)
    discovered_credentials: Set[str] = field(default_factory=set)
    network_topology: Dict[str, Any] = field(default_factory=dict)
    web_paths: Set[str] = field(default_factory=set)
    vulnerabilities: Set[str] = field(default_factory=set)
    shells_obtained: int = 0
    highest_privilege: str = "none"  # none, user, root
    attack_surface_score: float = 0.0

    def update_from_discoveries(self, discoveries: Dict[str, Any]) -> List[str]:
        """Update target model from discovery dict, return list of new findings."""
        new_findings = []
        for port in discoveries.get("ports", set()):
            if isinstance(port, (int, str)):
                port_num = int(port) if str(port).isdigit() else 0
                if port_num > 0 and port_num not in self.open_ports:
                    self.open_ports.add(port_num)
                    new_findings.append(f"port:{port_num}")
        for svc in discoveries.get("services", set()):
            if isinstance(svc, str) and svc not in {s.service for s in self.services.values()}:
                new_findings.append(f"service:{svc}")
        for cred in discoveries.get("credentials", set()):
            if isinstance(cred, str) and cred not in self.discovered_credentials:
                self.discovered_credentials.add(cred)
                new_findings.append(f"credential:{cred}")
        for user in discoveries.get("users", set()):
            if isinstance(user, str) and user not in self.discovered_users:
                self.discovered_users.add(user)
                new_findings.append(f"user:{user}")
        for vuln in discoveries.get("vulns", set()):
            if isinstance(vuln, str) and vuln not in self.vulnerabilities:
                self.vulnerabilities.add(vuln)
                new_findings.append(f"vuln:{vuln}")
        for path in discoveries.get("web_paths", set()):
            if isinstance(path, str) and path not in self.web_paths:
                self.web_paths.add(path)
                new_findings.append(f"web_path:{path}")
        shells = discoveries.get("shells", set())
        if shells and len(shells) > self.shells_obtained:
            self.shells_obtained = len(shells)
            new_findings.append(f"shells:{self.shells_obtained}")
        return new_findings

    def get_attack_surface_summary(self) -> str:
        """Generate a concise attack surface summary for LLM prompts."""
        lines = [f"Target: {self.ip} | OS: {self.os_guess} ({self.os_confidence:.0%})"]
        lines.append(f"Ports: {sorted(self.open_ports)[:20]} ({len(self.open_ports)} total)")
        if self.services:
            svc_str = ", ".join(f"{p}:{s.service}" for p, s in sorted(self.services.items())[:10])
            lines.append(f"Services: {svc_str}")
        if self.discovered_credentials:
            lines.append(f"Credentials: {len(self.discovered_credentials)} found")
        if self.vulnerabilities:
            lines.append(f"Vulns: {', '.join(list(self.vulnerabilities)[:5])}")
        lines.append(f"Shells: {self.shells_obtained} | Privilege: {self.highest_privilege}")
        return "\n".join(lines)


# ─────────────────────────── Reasoning Chain ───────────────────────────

@dataclass
class ReasoningTrace:
    """WHY/WHEN/HOW trace for a single decision."""
    agent_id: str
    step: int
    command: str
    why: str            # Why this command was chosen
    when_context: str   # What state triggered this choice
    how_execution: str  # How it maps to the attack plan
    expected_outcome: str
    actual_outcome: Optional[str] = None
    outcome_match: Optional[bool] = None  # Did outcome match expectation?
    alternative_considered: Optional[str] = None
    reasoning_source: str = "ppo"  # ppo, mentor, codex, venice, playbook

    def to_prompt_text(self) -> str:
        """Format for injection into LLM prompts."""
        text = f"[Step {self.step}] {self.agent_id}: {self.command}\n"
        text += f"  WHY: {self.why}\n"
        text += f"  CONTEXT: {self.when_context}\n"
        if self.actual_outcome:
            match_str = "✓" if self.outcome_match else "✗"
            text += f"  RESULT: {match_str} {self.actual_outcome[:100]}\n"
        return text


# ─────────────────────────── Venice Insight Bridge ───────────────────────────

@dataclass
class VeniceInsight:
    """A Venice GLM reasoning insight ready for cross-system consumption."""
    insight_type: str  # correlation, vulnerability, credential_reuse, chain, pattern
    content: str
    confidence: float
    discovered_at_step: int
    episode: int
    related_services: List[str] = field(default_factory=list)
    exploitation_hint: Optional[str] = None
    is_cross_episode: bool = False
    consumed_by: Set[str] = field(default_factory=set)  # Track which systems consumed this


class VeniceInsightBridge:
    """Bridges Venice GLM aha moments into Mentor/Codex/PPO prompts."""

    def __init__(self, max_insights: int = 100, max_per_prompt: int = 8):
        self._insights: deque = deque(maxlen=max_insights)
        self._cross_episode_insights: deque = deque(maxlen=50)
        self._max_per_prompt = max_per_prompt
        self._lock = threading.Lock()

    def add_insight(self, insight: VeniceInsight) -> None:
        """Add a Venice insight to the bridge."""
        with self._lock:
            self._insights.append(insight)
            if insight.is_cross_episode:
                self._cross_episode_insights.append(insight)
            logger.debug(f"Venice insight added: {insight.insight_type} (conf={insight.confidence:.2f})")

    def get_for_mentor(self, phase: str = "", max_items: int = 0) -> str:
        """Get Venice insights formatted for SmartMentor system prompt injection."""
        max_items = max_items or self._max_per_prompt
        with self._lock:
            relevant = self._rank_insights(phase)[:max_items]
            if not relevant:
                return ""
            lines = ["=== VENICE GLM REASONING INSIGHTS ==="]
            lines.append("These are AI-analyzed correlations and patterns from command outputs:")
            for ins in relevant:
                ins.consumed_by.add("mentor")
                lines.append(f"  [{ins.insight_type.upper()}] {ins.content}")
                if ins.exploitation_hint:
                    lines.append(f"    → Exploitation hint: {ins.exploitation_hint}")
            return "\n".join(lines)

    def get_for_codex(self, persona: str = "", phase: str = "") -> str:
        """Get Venice insights formatted for Codex persona prompts."""
        with self._lock:
            relevant = self._rank_insights(phase)[:5]
            if not relevant:
                return ""
            lines = ["[VENICE INTELLIGENCE]"]
            for ins in relevant:
                ins.consumed_by.add(f"codex_{persona}")
                lines.append(f"• {ins.content}")
            return "\n".join(lines)

    def get_for_ppo_context(self) -> Dict[str, float]:
        """Get Venice insight features as numerical context for PPO state encoding."""
        with self._lock:
            if not self._insights:
                return {"venice_insights_count": 0.0, "venice_avg_confidence": 0.0,
                        "venice_has_chain": 0.0, "venice_has_cred_reuse": 0.0}
            insights = list(self._insights)
            return {
                "venice_insights_count": min(len(insights) / 10.0, 1.0),
                "venice_avg_confidence": sum(i.confidence for i in insights) / len(insights),
                "venice_has_chain": 1.0 if any(i.insight_type == "chain" for i in insights) else 0.0,
                "venice_has_cred_reuse": 1.0 if any(i.insight_type == "credential_reuse" for i in insights) else 0.0,
            }

    def get_cross_episode_summary(self) -> str:
        """Get persistent cross-episode insights for pre-episode briefing."""
        with self._lock:
            if not self._cross_episode_insights:
                return ""
            lines = ["=== CROSS-EPISODE INTELLIGENCE (Venice) ==="]
            for ins in list(self._cross_episode_insights)[-10:]:
                lines.append(f"  [Ep{ins.episode}] {ins.content}")
            return "\n".join(lines)

    def _rank_insights(self, phase: str = "") -> List[VeniceInsight]:
        """Rank insights by relevance and recency."""
        insights = list(self._insights)
        scored = []
        for ins in insights:
            score = ins.confidence
            # Recency bonus
            age = len(self._insights) - list(self._insights).index(ins)
            score += max(0, 1.0 - age / 20.0) * 0.3
            # Cross-episode bonus
            if ins.is_cross_episode:
                score += 0.2
            # Unconsumed bonus
            if not ins.consumed_by:
                score += 0.3
            scored.append((score, ins))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ins for _, ins in scored]

    def reset_episode(self) -> None:
        """Reset episode-scoped insights, keep cross-episode."""
        with self._lock:
            self._insights.clear()


# ─────────────────────────── Learning Signal Aggregator ───────────────────────────

@dataclass
class AlgorithmPerformance:
    """Track per-algorithm performance for Orion optimization."""
    algorithm: str  # ppo, sac, ddqn, rnd
    agent_id: str
    episode: int
    decisions_made: int = 0
    positive_outcomes: int = 0
    negative_outcomes: int = 0
    avg_reward: float = 0.0
    total_reward: float = 0.0
    entropy: float = 0.0
    loss: float = 0.0
    gradient_norm: float = 0.0


class LearningSignalAggregator:
    """Aggregates learning signals for Orion post-episode optimization."""

    def __init__(self):
        self._performance: Dict[str, Dict[str, AlgorithmPerformance]] = defaultdict(dict)
        self._episode_history: deque = deque(maxlen=50)
        self._optimization_suggestions: List[str] = []
        self._lock = threading.Lock()

    def record_decision(self, agent_id: str, algorithm: str, episode: int,
                        reward: float, success: bool) -> None:
        """Record an algorithm's decision outcome."""
        with self._lock:
            key = f"{agent_id}:{algorithm}"
            if key not in self._performance:
                self._performance[key] = {}
            ep_key = str(episode)
            if ep_key not in self._performance[key]:
                self._performance[key][ep_key] = AlgorithmPerformance(
                    algorithm=algorithm, agent_id=agent_id, episode=episode
                )
            perf = self._performance[key][ep_key]
            perf.decisions_made += 1
            perf.total_reward += reward
            perf.avg_reward = perf.total_reward / perf.decisions_made
            if success:
                perf.positive_outcomes += 1
            else:
                perf.negative_outcomes += 1

    def record_update(self, agent_id: str, algorithm: str, episode: int,
                      loss: float, grad_norm: float, entropy: float) -> None:
        """Record gradient update statistics."""
        with self._lock:
            key = f"{agent_id}:{algorithm}"
            ep_key = str(episode)
            if key in self._performance and ep_key in self._performance[key]:
                perf = self._performance[key][ep_key]
                perf.loss = loss
                perf.gradient_norm = grad_norm
                perf.entropy = entropy

    def get_optimization_context(self, episode: int) -> str:
        """Generate optimization context for Orion GPT-5.2-codex analysis."""
        with self._lock:
            lines = ["=== ALGORITHM PERFORMANCE ANALYSIS ==="]
            ep_key = str(episode)
            for key, eps in self._performance.items():
                if ep_key in eps:
                    perf = eps[ep_key]
                    success_rate = (perf.positive_outcomes / max(perf.decisions_made, 1)) * 100
                    lines.append(
                        f"  {perf.agent_id}/{perf.algorithm}: "
                        f"{perf.decisions_made} decisions, "
                        f"{success_rate:.0f}% success, "
                        f"avg_reward={perf.avg_reward:.1f}, "
                        f"loss={perf.loss:.4f}, entropy={perf.entropy:.4f}"
                    )
            if self._optimization_suggestions:
                lines.append("\nPrior optimization suggestions:")
                for s in self._optimization_suggestions[-5:]:
                    lines.append(f"  • {s}")
            return "\n".join(lines)

    def add_optimization_suggestion(self, suggestion: str) -> None:
        """Add Orion's optimization suggestion for future reference."""
        with self._lock:
            self._optimization_suggestions.append(suggestion)
            logger.info(f"[ORION-OPT] {suggestion}")

    def get_trend_analysis(self, agent_id: str, algorithm: str, window: int = 5) -> Dict:
        """Get performance trend for a specific agent/algorithm pair."""
        with self._lock:
            key = f"{agent_id}:{algorithm}"
            if key not in self._performance:
                return {"trend": "unknown", "samples": 0}
            episodes = sorted(self._performance[key].keys(), key=int)[-window:]
            if len(episodes) < 2:
                return {"trend": "insufficient_data", "samples": len(episodes)}
            rewards = [self._performance[key][ep].avg_reward for ep in episodes]
            # Simple trend: positive if improving
            if len(rewards) >= 2:
                trend = "improving" if rewards[-1] > rewards[0] else "declining"
            else:
                trend = "stable"
            return {
                "trend": trend,
                "samples": len(episodes),
                "recent_avg": sum(rewards[-3:]) / len(rewards[-3:]) if rewards else 0,
                "overall_avg": sum(rewards) / len(rewards) if rewards else 0,
            }

    def reset_episode(self) -> None:
        """Archive current episode data."""
        pass  # Performance data persists across episodes for trend analysis


# ─────────────────────────── THE COGNITIVE BUS ───────────────────────────

class CognitiveBus:
    """
    The Unified Cognitive Bus — central nervous system of Ariaska_RL.

    All agents, algorithms, and LLM layers communicate through this bus.
    It maintains:
    1. A unified timeline of all events (actions, discoveries, reasoning)
    2. A shared belief state about the target (what we know)
    3. Reasoning traces (WHY/WHEN/HOW for every decision)
    4. Venice insight bridge (LLM analysis → all components)
    5. Learning signal aggregator (algorithm performance → Orion)

    Thread-safe. Singleton pattern enforced.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, max_timeline: int = 5000):
        if self._initialized:
            return
        self._initialized = True

        # Core components
        self.timeline: deque = deque(maxlen=max_timeline)
        self.target_model = TargetModel()
        self.venice_bridge = VeniceInsightBridge()
        self.learning_aggregator = LearningSignalAggregator()

        # Per-episode reasoning traces
        self._reasoning_traces: List[ReasoningTrace] = []
        self._inter_agent_messages: deque = deque(maxlen=200)

        # Cross-episode persistent state
        self._episode_summaries: deque = deque(maxlen=100)
        self._cross_episode_patterns: List[Dict[str, Any]] = []

        # Current episode tracking
        self._current_episode: int = 0
        self._current_step: int = 0
        self._episode_start_time: float = 0.0

        # Thread safety — RLock (reentrant) because end_episode()
        # calls get_episode_narrative() while holding the lock.
        self._bus_lock = threading.RLock()

        logger.info("CognitiveBus initialized (unified cognitive architecture)")

    # ─── Event Recording ───

    def record_action(self, agent_id: str, command: str, source: str,
                      reward: float, output_summary: str, reasoning: str = "",
                      confidence: float = 0.5, discoveries: Optional[Dict] = None) -> str:
        """Record an agent action with full context."""
        event = CognitiveEvent(
            timestamp=time.time(),
            event_type=EventType.ACTION,
            agent_id=agent_id,
            episode=self._current_episode,
            step=self._current_step,
            content={
                "command": command,
                "source": source,
                "reward": reward,
                "output_summary": output_summary[:300],
                "discoveries": discoveries or {},
            },
            reasoning=reasoning,
            confidence=confidence,
            source_algorithm=source,
        )
        with self._bus_lock:
            self.timeline.append(event)
            # Update target model with discoveries
            if discoveries:
                new_findings = self.target_model.update_from_discoveries(discoveries)
                if new_findings:
                    self._record_discovery_event(agent_id, new_findings)
            # Record learning signal
            success = reward > 0
            self.learning_aggregator.record_decision(
                agent_id, source, self._current_episode, reward, success
            )
        return event.event_id

    def record_reasoning(self, trace: ReasoningTrace) -> None:
        """Record a reasoning trace for a decision."""
        with self._bus_lock:
            self._reasoning_traces.append(trace)
            event = CognitiveEvent(
                timestamp=time.time(),
                event_type=EventType.REASONING,
                agent_id=trace.agent_id,
                episode=self._current_episode,
                step=trace.step,
                content={
                    "command": trace.command,
                    "why": trace.why,
                    "when": trace.when_context,
                    "how": trace.how_execution,
                    "expected": trace.expected_outcome,
                    "source": trace.reasoning_source,
                },
                reasoning=trace.why,
                source_algorithm=trace.reasoning_source,
            )
            self.timeline.append(event)

    def record_insight(self, agent_id: str, insight_type: str, content: str,
                       confidence: float = 0.5, source: str = "venice",
                       exploitation_hint: str = "", is_cross_episode: bool = False) -> None:
        """Record a Venice/Mentor insight and bridge it to all systems."""
        venice_insight = VeniceInsight(
            insight_type=insight_type,
            content=content,
            confidence=confidence,
            discovered_at_step=self._current_step,
            episode=self._current_episode,
            exploitation_hint=exploitation_hint,
            is_cross_episode=is_cross_episode,
        )
        self.venice_bridge.add_insight(venice_insight)

        event = CognitiveEvent(
            timestamp=time.time(),
            event_type=EventType.INSIGHT,
            agent_id=agent_id,
            episode=self._current_episode,
            step=self._current_step,
            content={
                "insight_type": insight_type,
                "text": content,
                "exploitation_hint": exploitation_hint,
            },
            confidence=confidence,
            source_algorithm=source,
            persistence="session" if is_cross_episode else "episode",
        )
        with self._bus_lock:
            self.timeline.append(event)

    def record_inter_agent_message(self, from_agent: str, to_agent: str,
                                    message_type: str, content: str) -> None:
        """Record inter-agent communication."""
        msg = {
            "from": from_agent,
            "to": to_agent,
            "type": message_type,
            "content": content,
            "episode": self._current_episode,
            "step": self._current_step,
            "timestamp": time.time(),
        }
        with self._bus_lock:
            self._inter_agent_messages.append(msg)
            event = CognitiveEvent(
                timestamp=time.time(),
                event_type=EventType.COMMUNICATION,
                agent_id=from_agent,
                episode=self._current_episode,
                step=self._current_step,
                content=msg,
            )
            self.timeline.append(event)

    def _record_discovery_event(self, agent_id: str, findings: List[str]) -> None:
        """Internal: record discovery events (called under lock)."""
        event = CognitiveEvent(
            timestamp=time.time(),
            event_type=EventType.DISCOVERY,
            agent_id=agent_id,
            episode=self._current_episode,
            step=self._current_step,
            content={"new_findings": findings},
            confidence=0.8,
        )
        self.timeline.append(event)

    # ─── Context Retrieval for LLM Prompts ───

    def get_mentor_context(self, agent_id: str = "", phase: str = "",
                           max_items: int = 15) -> str:
        """Get unified context for SmartMentor prompts.

        Combines:
        - Venice insights (aha moments, correlations)
        - Recent reasoning traces (WHY decisions were made)
        - Target model summary
        - Inter-agent communications
        - Cross-episode intelligence
        """
        sections = []

        # 1. Venice insights
        venice_ctx = self.venice_bridge.get_for_mentor(phase=phase)
        if venice_ctx:
            sections.append(venice_ctx)

        # 2. Cross-episode intelligence
        cross_ep = self.venice_bridge.get_cross_episode_summary()
        if cross_ep:
            sections.append(cross_ep)

        # 3. Target model
        if self.target_model.open_ports:
            sections.append("=== TARGET INTELLIGENCE ===")
            sections.append(self.target_model.get_attack_surface_summary())

        # 4. Recent reasoning from all agents (team awareness)
        with self._bus_lock:
            recent_traces = [t for t in self._reasoning_traces[-20:]
                             if t.agent_id != agent_id][-5:]
        if recent_traces:
            sections.append("=== TEAM REASONING (other agents) ===")
            for t in recent_traces:
                sections.append(t.to_prompt_text())

        # 5. Recent inter-agent messages for this agent
        with self._bus_lock:
            messages = [m for m in self._inter_agent_messages
                        if m.get("to") == agent_id or m.get("to") == "all"][-5:]
        if messages:
            sections.append("=== DIRECTIVES RECEIVED ===")
            for m in messages:
                sections.append(f"  [{m['from']}→{m['to']}] {m['content'][:150]}")

        return "\n\n".join(sections) if sections else ""

    def get_codex_context(self, agent_id: str = "", persona: str = "",
                          phase: str = "") -> str:
        """Get context for Codex persona prompts."""
        sections = []

        # Venice insights for codex
        venice_ctx = self.venice_bridge.get_for_codex(persona=persona, phase=phase)
        if venice_ctx:
            sections.append(venice_ctx)

        # Target model summary
        if self.target_model.open_ports:
            sections.append(f"[TARGET] {self.target_model.get_attack_surface_summary()}")

        # Learning aggregator trends
        trend_data = self.learning_aggregator.get_trend_analysis(agent_id, "ppo")
        if trend_data.get("trend") != "unknown":
            sections.append(f"[LEARNING] PPO trend: {trend_data['trend']}, "
                            f"recent_avg: {trend_data.get('recent_avg', 0):.1f}")

        return "\n".join(sections) if sections else ""

    def get_ppo_context_features(self) -> Dict[str, float]:
        """Get numerical features from cognitive bus for PPO state encoding."""
        features = self.venice_bridge.get_for_ppo_context()
        with self._bus_lock:
            features["reasoning_traces_count"] = min(len(self._reasoning_traces) / 20.0, 1.0)
            features["inter_agent_messages"] = min(len(self._inter_agent_messages) / 10.0, 1.0)
            features["target_ports_known"] = min(len(self.target_model.open_ports) / 20.0, 1.0)
            features["target_creds_found"] = min(len(self.target_model.discovered_credentials) / 5.0, 1.0)
            features["target_shells"] = min(self.target_model.shells_obtained / 3.0, 1.0)
        return features

    def get_episode_narrative(self, max_events: int = 50) -> str:
        """Get a narrative summary of the episode for Orion postmortem."""
        with self._bus_lock:
            events = [e for e in self.timeline
                      if e.episode == self._current_episode][-max_events:]
        if not events:
            return "No events recorded this episode."
        lines = [f"Episode {self._current_episode} Narrative ({len(events)} events):"]
        for e in events:
            if e.event_type == EventType.ACTION:
                cmd = e.content.get("command", "?")[:60]
                reward = e.content.get("reward", 0)
                lines.append(f"  [s{e.step}] {e.agent_id} → {cmd} (R:{reward:+.1f})")
                if e.reasoning:
                    lines.append(f"         WHY: {e.reasoning[:100]}")
            elif e.event_type == EventType.DISCOVERY:
                findings = e.content.get("new_findings", [])
                if findings:
                    lines.append(f"  [s{e.step}] DISCOVERY: {', '.join(findings[:5])}")
            elif e.event_type == EventType.INSIGHT:
                lines.append(f"  [s{e.step}] INSIGHT ({e.content.get('insight_type', '?')}): "
                             f"{e.content.get('text', '')[:100]}")
            elif e.event_type == EventType.PHASE_CHANGE:
                lines.append(f"  [s{e.step}] *** PHASE → {e.content.get('new_phase', '?')} ***")
        return "\n".join(lines)

    # ─── Episode Lifecycle ───

    def start_episode(self, episode: int, target_ip: str = "") -> None:
        """Initialize for a new episode."""
        with self._bus_lock:
            self._current_episode = episode
            self._current_step = 0
            self._episode_start_time = time.time()
            self._reasoning_traces.clear()
            self.target_model = TargetModel(ip=target_ip)
            self.venice_bridge.reset_episode()
            self.learning_aggregator.reset_episode()
            # Keep inter-agent messages and cross-episode data
        logger.info(f"CognitiveBus: Episode {episode} started (target={target_ip})")

    def advance_step(self, step: int) -> None:
        """Update current step counter."""
        self._current_step = step

    def record_phase_change(self, old_phase: str, new_phase: str, agent_id: str = "") -> None:
        """Record a phase transition."""
        event = CognitiveEvent(
            timestamp=time.time(),
            event_type=EventType.PHASE_CHANGE,
            agent_id=agent_id or "system",
            episode=self._current_episode,
            step=self._current_step,
            content={"old_phase": old_phase, "new_phase": new_phase},
            confidence=1.0,
        )
        with self._bus_lock:
            self.timeline.append(event)

    def end_episode(self) -> Dict[str, Any]:
        """Finalize episode and return summary for Orion analysis."""
        with self._bus_lock:
            duration = time.time() - self._episode_start_time
            ep_events = [e for e in self.timeline if e.episode == self._current_episode]
            action_events = [e for e in ep_events if e.event_type == EventType.ACTION]
            total_reward = sum(e.content.get("reward", 0) for e in action_events)
            unique_commands = len(set(e.content.get("command", "") for e in action_events))

            summary = {
                "episode": self._current_episode,
                "duration": duration,
                "total_events": len(ep_events),
                "total_actions": len(action_events),
                "total_reward": total_reward,
                "unique_commands": unique_commands,
                "discoveries": len([e for e in ep_events if e.event_type == EventType.DISCOVERY]),
                "insights": len([e for e in ep_events if e.event_type == EventType.INSIGHT]),
                "reasoning_traces": len(self._reasoning_traces),
                "inter_agent_messages": len([m for m in self._inter_agent_messages
                                              if m.get("episode") == self._current_episode]),
                "target_model": {
                    "ports": len(self.target_model.open_ports),
                    "services": len(self.target_model.services),
                    "credentials": len(self.target_model.discovered_credentials),
                    "shells": self.target_model.shells_obtained,
                    "privilege": self.target_model.highest_privilege,
                },
                "algorithm_performance": self.learning_aggregator.get_optimization_context(
                    self._current_episode
                ),
                "narrative": self.get_episode_narrative(max_events=30),
            }
            self._episode_summaries.append(summary)
            return summary

    # ─── Singleton Reset (for testing) ───

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset the singleton instance (for tests only)."""
        with cls._lock:
            cls._instance = None


def get_cognitive_bus() -> CognitiveBus:
    """Get or create the global CognitiveBus instance."""
    return CognitiveBus()
