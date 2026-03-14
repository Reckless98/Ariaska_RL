#!/usr/bin/env python3
"""
core/execution/llm_output_interpreter.py — LLM Output Interpreter

LLM-based interpretation that teaches agents to reason about command output:

  Primary: local-llm — deep reasoning, autonomous interpretation, high-quality discovery extraction

The interpreter doesn't just extract discoveries — it produces an
"interpretation lesson" that agents can learn from: WHY certain output
patterns indicate specific discoveries, HOW to chain findings, and
WHAT to look for next.

Author: Filip Volf / Ariaska System
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.gpt_manager import GPTManager

logger = logging.getLogger("ariaska.llm_output_interpreter")

# ── Model configuration ─────────────────────────────────────────────
LOCAL_LLM_MODEL = "local-llm"


@dataclass
class InterpretationLesson:
    """
    A structured interpretation lesson that teaches agents how to
    understand command output and reason about discoveries.

    This is the KEY learning artifact — agents store these and use them
    to improve future output interpretation without LLM calls.
    """
    # What was found
    discoveries: Dict[str, Any] = field(default_factory=dict)

    # Teaching content — WHY and HOW
    reasoning: str = ""          # Why does this output indicate these discoveries?
    patterns_learned: List[str] = field(default_factory=list)  # Regex/keyword patterns the agent should learn
    next_steps: List[str] = field(default_factory=list)       # What to do with these discoveries
    interpretation_chain: str = ""  # Step-by-step reasoning chain

    # Phase 23: GPT-driven phase assessment
    phase_assessment: Dict[str, Any] = field(default_factory=dict)  # {current_phase_complete, ready_for_next, reasoning, recommended_next_phase}
    use_regex_hint: bool = False  # GPT says this output type is simple enough for regex

    # Meta
    model_used: str = ""
    provider: str = ""           # "local"
    confidence: float = 0.0
    latency_ms: float = 0.0
    tokens_used: int = 0
    command: str = ""
    agent_name: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_learning_context(self) -> str:
        """
        Produce a compact string an agent can store in its reasoning memory
        for future steps — teaching it to interpret similar output later.
        """
        parts = []
        if self.reasoning:
            parts.append(f"Lesson: {self.reasoning[:200]}")
        if self.patterns_learned:
            parts.append(f"Patterns: {'; '.join(self.patterns_learned[:5])}")
        if self.next_steps:
            parts.append(f"Next: {'; '.join(self.next_steps[:3])}")
        return " | ".join(parts) if parts else ""


class LLMOutputInterpreter:
    """
    LLM-driven output interpreter that teaches agents to reason about
    command output.

    Pipeline:
      1. Local LLM interprets output (primary — deep reasoning)
      2. Extract structured discoveries + interpretation lesson
      3. Feed lesson back to agent's reasoning memory

    The interpreter is designed for the "intelligent_fullparse" mode.
    In "fast" mode, it's skipped entirely — regex handles those.
    """

    def __init__(
        self,
        gpt_manager: Optional["GPTManager"] = None,
        max_gpt_calls_per_episode: int = 45,
        min_output_length: int = 30,
    ):
        self._gpt = gpt_manager
        self._max_gpt = max_gpt_calls_per_episode
        self._min_output_len = min_output_length

        # Per-episode counters
        self._gpt_calls: int = 0

        # Cross-episode learned patterns — agents accumulate these
        self._learned_patterns: Dict[str, List[str]] = {}  # agent_name -> [pattern strings]
        self._max_patterns_per_agent = 50

        # Stats
        self._stats = {
            "total_calls": 0,
            "llm_calls": 0,
            "llm_successes": 0,
            "lessons_produced": 0,
            "patterns_learned": 0,
            "total_tokens": 0,
        }

    def reset_episode(self) -> None:
        """Reset per-episode call counters."""
        self._gpt_calls = 0

    def can_interpret(self) -> bool:
        """Check if we have budget for another interpretation call."""
        return self._llm_available()

    def _llm_available(self) -> bool:
        """Check if local LLM is available and within budget."""
        if not self._gpt or self._gpt.is_offline():
            return False
        return self._gpt_calls < self._max_gpt

    def interpret(
        self,
        command: str,
        output: str,
        agent_name: str = "unknown",
        phase: str = "RECON",
        known_discoveries: Optional[Dict[str, Any]] = None,
    ) -> InterpretationLesson:
        """
        Local LLM output interpretation with full reasoning.

        Produces InterpretationLessons that teach agents:
        - HOW to parse output (patterns, keywords, structure)
        - WHY certain output indicates specific discoveries
        - WHAT to do next based on findings

        Args:
            command: The command that was executed
            output: Raw command output
            agent_name: Which agent ran this command
            phase: Current attack phase (RECON, EXPLOITATION, etc.)
            known_discoveries: Previously known discoveries (avoid duplication)

        Returns:
            InterpretationLesson with discoveries and agent learning content
        """
        self._stats["total_calls"] += 1

        if not output or len(output.strip()) < self._min_output_len:
            return InterpretationLesson(command=command, agent_name=agent_name)

        # Skip trivial outputs (SIM markers, pure errors)
        if self._is_trivial(output):
            return InterpretationLesson(command=command, agent_name=agent_name)

        # Build the interpretation prompt
        prompt = self._build_prompt(command, output, agent_name, phase, known_discoveries)

        if self._llm_available():
            lesson = self._call_local_llm(prompt, command, agent_name)
            if lesson and lesson.discoveries:
                self._record_learned_patterns(agent_name, lesson)
                return lesson

        return InterpretationLesson(command=command, agent_name=agent_name)

    def _build_prompt(
        self,
        command: str,
        output: str,
        agent_name: str,
        phase: str,
        known_discoveries: Optional[Dict[str, Any]],
    ) -> str:
        """Build the interpretation + teaching prompt with phase readiness assessment."""
        # Truncate output to keep tokens reasonable
        max_chars = 2000
        truncated = output[:max_chars] if len(output) > max_chars else output

        # Include previously learned patterns for this agent
        _patterns_ctx = ""
        agent_patterns = self._learned_patterns.get(agent_name, [])
        if agent_patterns:
            _patterns_ctx = (
                "\n\nPATTERNS THIS AGENT HAS LEARNED:\n"
                + "\n".join(f"  - {p}" for p in agent_patterns[-10:])
            )

        # Known discoveries context
        _known_ctx = ""
        if known_discoveries:
            _known_parts = []
            for key, val in known_discoveries.items():
                if val and isinstance(val, (set, list)):
                    _known_parts.append(f"  {key}: {', '.join(str(v) for v in list(val)[:8])}")
                elif val:
                    _known_parts.append(f"  {key}: {val}")
            if _known_parts:
                _known_ctx = "\n\nALREADY KNOWN (don't repeat):\n" + "\n".join(_known_parts)

        return f"""You are an expert penetration tester teaching an AI agent ({agent_name}) to interpret command output.
The agent is in phase: {phase}

COMMAND: {command}

OUTPUT:
```
{truncated}
```
{_known_ctx}{_patterns_ctx}

Respond in JSON with these fields:
{{
    "discoveries": {{
        "open_port": [port numbers found],
        "service": ["service names"],
        "version_info": ["service version strings"],
        "credential": true/false,
        "user": ["usernames"],
        "vulnerability": true/false,
        "cve": ["CVE-XXXX-XXXXX"],
        "shell": true/false,
        "root_shell": true/false,
        "web_path": ["/paths"],
        "flag": true/false,
        "user_flag": true/false,
        "root_flag": true/false
    }},
    "reasoning": "WHY this output indicates these discoveries - explain the indicators",
    "patterns_learned": [
        "Pattern: 'XX/tcp open YY' means port XX runs service YY",
        "Pattern: 'login: username' in telnet output means credential available"
    ],
    "next_steps": [
        "specific command to run next based on what was found"
    ],
    "interpretation_chain": "Step 1: I see X in the output. Step 2: This means Y. Step 3: Therefore we should Z.",
    "phase_assessment": {{
        "current_phase_complete": true/false,
        "ready_for_next_phase": true/false,
        "reasoning": "Why the current phase is/isn't complete based on what we've found so far",
        "recommended_next_phase": "ENUMERATION or null if should stay"
    }},
    "use_regex_hint": true/false,
    "confidence": 0.0-1.0
}}

RULES:
- Only report NEW discoveries not in ALREADY KNOWN
- patterns_learned: teach the agent regex/keyword patterns to recognize in future output
- interpretation_chain: show step-by-step reasoning so the agent learns HOW to interpret
- phase_assessment: assess whether enough evidence has been gathered to advance the kill chain phase
- use_regex_hint: if true, this type of output is simple enough for regex parsing (saves tokens next time)
- Be concise but educational — the agent will memorize this for future steps
- If nothing useful: return {{"discoveries": {{}}, "reasoning": "No actionable findings", "confidence": 0.1}}

JSON:"""

    def _call_gpt_fallback(
        self,
        prompt: str,
        command: str,
        agent_name: str,
    ) -> Optional[InterpretationLesson]:
        """Fallback to local-llm when Venice is unavailable."""
        if not self._gpt:
            return None

        self._gpt_calls += 1
        self._stats["gpt_fallback_calls"] += 1
        t0 = time.time()

        try:
            response = self._gpt.gpt_request(
                prompt=prompt,
                task_type="reasoning",
                agent_id=f"interpreter_{agent_name}",
                max_tokens=750,  # Phase 11.5: +50% (was 500) for deeper local-llm reasoning
                model=GPT_FALLBACK_MODEL,
            )

            latency_ms = (time.time() - t0) * 1000

            if response:
                lesson = self._parse_response(response, command, agent_name)
                lesson.model_used = GPT_FALLBACK_MODEL
                lesson.provider = "gpt_fallback"
                lesson.latency_ms = latency_ms
                if lesson.discoveries:
                    self._stats["gpt_fallback_successes"] += 1
                    self._stats["lessons_produced"] += 1
                    logger.debug(
                        f"[LLM-INTERP] local-llm | {agent_name} | "
                        f"found={list(lesson.discoveries.keys())} | "
                        f"{latency_ms:.0f}ms"
                    )
                return lesson

        except Exception as e:
            logger.debug(f"[LLM-INTERP] GPT fallback error: {e}")

        return None

    def _teacher_validate(
        self,
        apprentice_lesson: InterpretationLesson,
        command: str,
        output: str,
        agent_name: str,
    ) -> Optional[InterpretationLesson]:
        """Phase 12.1: Teacher (GPT-5.2-codex) validates Apprentice (Venice) high-value discoveries.
        
        When Venice finds shells, credentials, or flags, GPT-5.2-codex double-checks
        the output to prevent false positives from inflating rewards and corrupting
        the reward signal. If Teacher disagrees, returns a corrected lesson.
        
        Args:
            apprentice_lesson: The lesson from Venice (Apprentice)
            command: Original command
            output: Original output
            agent_name: Agent name
            
        Returns:
            Corrected lesson if Teacher disagreed, None if Apprentice was correct
        """
        if not self._gpt:
            return None
        
        self._gpt_calls += 1
        self._stats["gpt_fallback_calls"] += 1
        t0 = time.time()
        
        # Build compact validation prompt
        truncated = output[:1500] if len(output) > 1500 else output
        apprentice_claims = json.dumps(apprentice_lesson.discoveries, default=str)
        
        validation_prompt = (
            f"VALIDATE these penetration testing discoveries.\n"
            f"Command: {command[:200]}\n"
            f"Output:\n{truncated[:800]}\n\n"
            f"Apprentice claims: {apprentice_claims}\n\n"
            f"Reply in JSON: {{\"valid\": true/false, \"corrections\": {{...}}, "
            f"\"reasoning\": \"why\"}}\n"
            f"If valid, return {{\"valid\": true}}. If false positives detected, "
            f"return corrected discoveries in corrections field."
        )
        
        try:
            response = self._gpt.gpt_request(
                prompt=validation_prompt,
                task_type="reasoning",
                agent_id=f"teacher_validate_{agent_name}",
                max_tokens=300,
                model=GPT_FALLBACK_MODEL,
            )
            
            latency_ms = (time.time() - t0) * 1000
            
            if response:
                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    if data.get("valid", True):
                        # Teacher agrees — Apprentice was correct
                        logger.debug(
                            f"[TEACHER] Validated {agent_name} discoveries "
                            f"({latency_ms:.0f}ms)"
                        )
                        return None
                    else:
                        # Teacher disagrees — use corrected discoveries
                        corrections = data.get("corrections", {})
                        if corrections:
                            corrected = InterpretationLesson(
                                command=command,
                                agent_name=agent_name,
                                discoveries=corrections,
                                reasoning=(
                                    f"Teacher corrected: {data.get('reasoning', 'false positive detected')}"
                                ),
                                patterns_learned=apprentice_lesson.patterns_learned,
                                next_steps=apprentice_lesson.next_steps,
                                model_used=GPT_FALLBACK_MODEL,
                                provider="teacher_validation",
                                confidence=0.85,
                                latency_ms=latency_ms + apprentice_lesson.latency_ms,
                                tokens_used=apprentice_lesson.tokens_used,
                            )
                            logger.debug(
                                f"[TEACHER] Corrected {agent_name}: "
                                f"{list(apprentice_lesson.discoveries.keys())} → "
                                f"{list(corrections.keys())} ({latency_ms:.0f}ms)"
                            )
                            return corrected
        except Exception as e:
            logger.debug(f"[TEACHER] Validation error: {e}")
        
        return None  # On error, trust Apprentice

    def _parse_response(
        self,
        response: str,
        command: str,
        agent_name: str,
    ) -> InterpretationLesson:
        """Parse LLM JSON response into an InterpretationLesson."""
        lesson = InterpretationLesson(command=command, agent_name=agent_name)

        try:
            # Extract JSON from response (may have markdown fences)
            json_text = response
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                # Try bare JSON object
                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_match:
                    json_text = json_match.group()

            data = json.loads(json_text)

            # Extract discoveries
            raw_disc = data.get("discoveries", {})
            if isinstance(raw_disc, dict):
                # Clean up: remove empty/null/false values
                lesson.discoveries = {
                    k: v for k, v in raw_disc.items()
                    if v and v is not False and v != [] and v != ""
                }

            # Extract teaching content
            lesson.reasoning = str(data.get("reasoning", ""))[:500]
            lesson.interpretation_chain = str(data.get("interpretation_chain", ""))[:500]
            lesson.confidence = float(data.get("confidence", 0.5))

            # Extract learned patterns
            patterns = data.get("patterns_learned", [])
            if isinstance(patterns, list):
                lesson.patterns_learned = [str(p)[:200] for p in patterns[:10]]

            # Extract next steps
            next_steps = data.get("next_steps", [])
            if isinstance(next_steps, list):
                lesson.next_steps = [str(s)[:200] for s in next_steps[:5]]

            # Phase 23: Extract phase assessment from GPT
            phase_assess = data.get("phase_assessment", {})
            if isinstance(phase_assess, dict):
                lesson.phase_assessment = phase_assess

            # Phase 23: Extract regex hint — GPT decides if regex could handle this
            lesson.use_regex_hint = bool(data.get("use_regex_hint", False))

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.debug(f"[LLM-INTERP] JSON parse error: {e}")
            # Try compact format fallback (PORT:num, SERVICE:name, ...)
            lesson.discoveries = self._parse_compact_fallback(response)
            lesson.reasoning = "Parsed from compact format (JSON parse failed)"

        return lesson

    def _parse_compact_fallback(self, response: str) -> Dict[str, Any]:
        """Parse compact discovery format when JSON fails."""
        discoveries: Dict[str, Any] = {}

        for item in response.split(","):
            item = item.strip()
            if item.startswith("PORT:"):
                try:
                    port = int(item.split(":")[1].strip())
                    discoveries.setdefault("open_port", []).append(port)
                except (ValueError, IndexError):
                    pass
            elif item.startswith("SERVICE:"):
                svc = item.split(":", 1)[1].strip().lower()
                if svc:
                    discoveries.setdefault("service", []).append(svc)
            elif item.startswith("CRED:"):
                discoveries["credential"] = True
            elif item.startswith("SHELL:"):
                discoveries["shell"] = True
                shell_type = item.split(":", 1)[1].strip().lower()
                if "root" in shell_type:
                    discoveries["root_shell"] = True
            elif item.startswith("CVE:"):
                cve_id = item.split(":", 1)[1].strip().upper()
                if cve_id.startswith("CVE-"):
                    discoveries.setdefault("cve", []).append(cve_id)
                    discoveries["vulnerability"] = True

        return discoveries

    def _record_learned_patterns(
        self, agent_name: str, lesson: InterpretationLesson
    ) -> None:
        """Store learned patterns for an agent — cross-episode memory."""
        if not lesson.patterns_learned:
            return

        if agent_name not in self._learned_patterns:
            self._learned_patterns[agent_name] = []

        existing = set(self._learned_patterns[agent_name])
        for p in lesson.patterns_learned:
            if p not in existing:
                self._learned_patterns[agent_name].append(p)
                self._stats["patterns_learned"] += 1

        # Cap patterns per agent
        if len(self._learned_patterns[agent_name]) > self._max_patterns_per_agent:
            self._learned_patterns[agent_name] = (
                self._learned_patterns[agent_name][-self._max_patterns_per_agent:]
            )

    def get_learned_patterns(self, agent_name: str) -> List[str]:
        """Get patterns this agent has learned from past interpretations."""
        return self._learned_patterns.get(agent_name, [])

    def get_all_learned_context(self, agent_name: str) -> str:
        """
        Get a compact summary of all patterns this agent has learned,
        suitable for injection into mentor/reasoning prompts.
        """
        patterns = self.get_learned_patterns(agent_name)
        if not patterns:
            return ""
        recent = patterns[-15:]  # Most recent patterns
        return (
            "OUTPUT INTERPRETATION LESSONS LEARNED:\n"
            + "\n".join(f"  • {p}" for p in recent)
        )

    @staticmethod
    def _is_trivial(output: str) -> bool:
        """Check if output is trivial (not worth LLM interpretation)."""
        stripped = output.strip()
        if stripped.startswith("[SIM]") and len(stripped) < 50:
            return True
        if len(stripped.split("\n")) <= 1 and len(stripped) < 30:
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get interpreter statistics."""
        return dict(self._stats)
