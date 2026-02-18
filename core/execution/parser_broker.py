#!/usr/bin/env python3
"""
core/execution/parser_broker.py — Phase 13.0: Intelligent Parser Broker v4.0

Dual-mode discovery parsing pipeline with GPT-primary LLM interpretation:

  MODE 1: "fast" (default)
    Stage 1: Regex (OutputParser) — free, handles 80%+
    Skips LLM stages entirely. Fast, deterministic.

  MODE 2: "intelligent_fullparse"
    Stage 1: Regex           — free, pattern matching
    Stage 2: SOP LLM         — nano-LLM fallback
    Stage 3: LLM Interpreter — GPT-5.2-codex primary / Venice qwen3-coder fallback
    Stage 4: GPT finaliser   — gpt-5-nano for edge cases

Phase 13.0: Stage 3 now uses GPT-5.2-codex as PRIMARY interpreter:
  - Uses gpt-5.2-codex for deep reasoning interpretation (primary)
  - Venice qwen3-coder validates high-value GPT discoveries (shells, creds, flags)
  - Falls back to Venice when GPT budget exhausted
  - Produces InterpretationLessons that teach agents to interpret output
  - Agents accumulate learned patterns cross-episode

Each stage only fires if prior stages found nothing meaningful.
All stages emit DiscoveryEvent objects with ParseExplanation annotations.

Author: Filip Volf / Ariaska System — Phase 10.0 → Phase 13.0
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from core.execution.discovery_event import DiscoveryEvent, DiscoveryType

if TYPE_CHECKING:
    from core.gpt_manager import GPTManager
    from core.telemetry.unified_trace import ParseExplanation
    from core.execution.parser_teacher_output import ParserTeacherOutput

logger = logging.getLogger("ariaska.parser_broker")


class ParserBroker:
    """
    Dual-mode output parsing pipeline (v4.0).

    Modes:
      "fast" — regex only, zero LLM calls, ~1ms latency
      "intelligent_fullparse" — 4-stage cascade with LLM interpretation + agent learning

    Phase 13.0: Stage 3 uses LLMOutputInterpreter (GPT-5.2-codex primary → Venice fallback)
    which teaches agents how to interpret command output via InterpretationLessons.
    Venice validates high-value GPT discoveries to prevent hallucinations.

    Usage:
        broker = ParserBroker(gpt_manager=gpt, venice=venice_layer)
        events, explanations = broker.parse(command, output, agent_name)
        flat = DiscoveryEvent.to_flat_discoveries(events)  # backward compat
    """

    def __init__(
        self,
        gpt_manager: Optional["GPTManager"] = None,
        venice: Optional[Any] = None,
        enable_venice: bool = True,
        enable_gpt: bool = True,
        max_llm_calls_per_episode: int = 20,
        max_venice_calls_per_episode: int = 15,
        default_mode: str = "fast",
    ):
        self._gpt = gpt_manager
        self._venice = venice
        self._enable_venice = enable_venice and venice is not None
        self._enable_gpt = enable_gpt and gpt_manager is not None
        self._max_llm_calls = max_llm_calls_per_episode
        self._max_venice_calls = max_venice_calls_per_episode
        self._default_mode = default_mode
        self._llm_calls: int = 0
        self._venice_calls: int = 0

        # Load SmartOutputParser (wraps Stage 1 regex + Stage 2 LLM)
        self._sop: Optional[Any] = None
        try:
            from core.execution.smart_output_parser import SmartOutputParser
            self._sop = SmartOutputParser(
                gpt_manager=gpt_manager,
                enable_llm=enable_gpt,
                max_llm_calls_per_episode=max_llm_calls_per_episode,
            )
        except ImportError:
            logger.warning("SmartOutputParser not available")

        # Phase 11.3: LLM Output Interpreter (qwen3-coder → gpt-5.2-mini)
        # This is the new stage that teaches agents to interpret output
        self._llm_interpreter: Optional[Any] = None
        try:
            from core.execution.llm_output_interpreter import LLMOutputInterpreter
            self._llm_interpreter = LLMOutputInterpreter(
                gpt_manager=gpt_manager,
                max_venice_calls_per_episode=max_venice_calls_per_episode,
                max_gpt_calls_per_episode=max_llm_calls_per_episode,
            )
            logger.info("[BROKER-11.3] LLMOutputInterpreter loaded (qwen3-coder → gpt-5.2-mini)")
        except Exception as e:
            logger.warning(f"LLMOutputInterpreter not available: {e}")

        # Stats
        self._stats = {
            "total_calls": 0,
            "stage1_hits": 0,    # regex
            "stage2_hits": 0,    # SOP LLM
            "stage3_hits": 0,    # LLM Interpreter (Venice qwen3-coder / gpt-5.2-mini)
            "stage4_hits": 0,    # GPT finaliser
            "empty_outputs": 0,
            "total_events": 0,
            "fast_calls": 0,
            "fullparse_calls": 0,
            "lessons_produced": 0,  # Phase 11.3: interpretation lessons generated
        }

        # Phase 11.3: Store last interpretation lesson for agent feedback
        self._last_lesson: Optional[Any] = None

    def reset_episode(self) -> None:
        """Reset per-episode counters."""
        self._llm_calls = 0
        self._venice_calls = 0
        if self._sop:
            self._sop.reset_episode()
        # Phase 11.3: Reset LLM interpreter episode counters
        if self._llm_interpreter:
            self._llm_interpreter.reset_episode()

    def parse(
        self,
        command: str,
        output: str,
        agent_name: str = "unknown",
        mode: Optional[str] = None,
    ) -> List[DiscoveryEvent]:
        """
        Run the parsing pipeline and return DiscoveryEvents.

        Args:
            command: The command that was executed
            output: Raw command output text
            agent_name: Which agent ran this
            mode: Override parse mode ("fast" or "intelligent_fullparse")

        Returns:
            List of DiscoveryEvent objects
        """
        effective_mode = mode or self._default_mode
        self._stats["total_calls"] += 1

        if not output or len(output.strip()) < 5:
            self._stats["empty_outputs"] += 1
            return []

        # Phase 18: Strip ANSI escape codes from raw output before any
        # regex stage — real tools emit colour/cursor sequences.
        import re as _re
        output = _re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', output)
        output = output.replace('\r', '')

        if effective_mode == "intelligent_fullparse":
            self._stats["fullparse_calls"] += 1
            return self._parse_fullparse(command, output, agent_name)
        else:
            self._stats["fast_calls"] += 1
            return self._parse_fast(command, output, agent_name)

    def parse_as_teacher(
        self,
        command: str,
        output: str,
        agent_name: str = "unknown",
        mode: Optional[str] = None,
        known_discoveries: Optional[set] = None,
    ) -> "ParserTeacherOutput":
        """
        Phase 14.0: Parse and return ParserTeacherOutput with learning features.

        Feature-flag gated by ff.parser_teacher. Falls back to wrapping
        standard parse() events when the flag is enabled.
        """
        import time as _t
        t0 = _t.time()
        events = self.parse(command, output, agent_name, mode=mode)
        latency_ms = (_t.time() - t0) * 1000

        from core.execution.parser_teacher_output import (
            ParserTeacherOutput, LearningFeatures, ParsingLesson,
        )

        # Build learning features
        discovery_counts: Dict[str, int] = {}
        confidences = []
        novel = 0
        repeated = 0
        _known = known_discoveries or set()

        for ev in events:
            dt_val = ev.discovery_type.value if hasattr(ev.discovery_type, 'value') else str(ev.discovery_type)
            discovery_counts[dt_val] = discovery_counts.get(dt_val, 0) + 1
            confidences.append(getattr(ev, 'confidence', 1.0))
            ev_key = f"{dt_val}:{getattr(ev, 'value', '')}"
            if ev_key in _known:
                repeated += 1
            else:
                novel += 1

        stage_reached = 0
        for ev in events:
            s = {"regex": 1, "sop_llm": 2, "venice": 3, "gpt_finaliser": 4}.get(
                getattr(ev, 'source_stage', ''), 0
            )
            stage_reached = max(stage_reached, s)

        lf = LearningFeatures(
            discovery_counts=discovery_counts,
            confidence_mean=sum(confidences) / len(confidences) if confidences else 0.0,
            confidence_min=min(confidences) if confidences else 0.0,
            stage_reached=stage_reached,
            novel_discoveries=novel,
            repeated_discoveries=repeated,
            total_events=len(events),
            parser_latency_ms=latency_ms,
        )

        # Build lessons from patterns
        lessons = []
        seen_types = set()
        for ev in events[:3]:
            dt_val = ev.discovery_type.value if hasattr(ev.discovery_type, 'value') else str(ev.discovery_type)
            if dt_val not in seen_types:
                seen_types.add(dt_val)
                lessons.append(ParsingLesson(
                    pattern=getattr(ev, 'source_stage', 'regex'),
                    discovery_type=dt_val,
                    confidence=getattr(ev, 'confidence', 1.0),
                    example_output=str(getattr(ev, 'raw_evidence', ''))[:128],
                ))

        return ParserTeacherOutput(
            events=events,
            learning_features=lf,
            lessons=lessons,
        )

    def parse_with_explanations(
        self,
        command: str,
        output: str,
        agent_name: str = "unknown",
        mode: Optional[str] = None,
    ) -> tuple:
        """
        Parse and return (events, explanations, latency_ms, stage_reached).

        This is the v2.0 API used by UnifiedStepTrace building.
        Returns a tuple of:
          - List[DiscoveryEvent]
          - List[ParseExplanation]
          - float (total latency in ms)
          - int (highest stage number reached, 1-4)
        """
        from core.telemetry.unified_trace import ParseExplanation

        effective_mode = mode or self._default_mode
        t0 = time.time()

        events = self.parse(command, output, agent_name, mode=effective_mode)

        latency_ms = (time.time() - t0) * 1000

        # Build explanations from events
        explanations = []
        stage_reached = 0
        for ev in events:
            stage_num = {"regex": 1, "sop_llm": 2, "venice": 3, "gpt_finaliser": 4}.get(
                ev.source_stage, 0
            )
            stage_reached = max(stage_reached, stage_num)
            explanations.append(ParseExplanation(
                stage=ev.source_stage,
                stage_number=stage_num,
                discovery_type=ev.discovery_type.value if ev.discovery_type else "",
                discovery_value=str(ev.value),
                confidence=ev.confidence,
                reasoning=f"Stage {stage_num} ({ev.source_stage}) found {ev.discovery_type.value}: {ev.value}",
                raw_match=ev.raw_evidence[:200] if ev.raw_evidence else "",
                latency_ms=latency_ms / max(len(events), 1),
            ))

        return events, explanations, latency_ms, stage_reached

    def _parse_fast(
        self,
        command: str,
        output: str,
        agent_name: str,
    ) -> List[DiscoveryEvent]:
        """Fast mode: regex only, zero LLM calls."""
        flat_discoveries = {}
        source_stage = "regex"

        if self._sop:
            # Use SOP but only the regex path (disable LLM)
            flat_discoveries = self._sop.parse(
                command=command,
                output=output,
                agent_name=agent_name,
            )
            if flat_discoveries:
                self._stats["stage1_hits"] += 1

        # Phase 19: Ultra-smart regex patterns for CrushFTP, Erlang, PCAP, vhost
        if not flat_discoveries:
            flat_discoveries = self._phase19_regex_extract(command, output)
            if flat_discoveries:
                self._stats["stage1_hits"] += 1
                source_stage = "regex_p19"

        events = DiscoveryEvent.from_flat_discoveries(
            discoveries=flat_discoveries,
            source_stage=source_stage,
            command=command,
            agent=agent_name,
        )
        self._stats["total_events"] += len(events)
        return events

    def _phase19_regex_extract(
        self,
        command: str,
        output: str,
    ) -> Dict[str, Any]:
        """Phase 19: Ultra-smart regex extraction for CrushFTP, Erlang, PCAP, vhost patterns."""
        import re
        discoveries: Dict[str, Any] = {}
        output_lower = output.lower()

        # CrushFTP auth bypass / user list
        if "getuserlist" in output_lower or "crushftp" in output_lower:
            usernames = re.findall(r'<username>(\w+)</username>', output)
            if usernames:
                discoveries["credential"] = True
                discoveries.setdefault("users", []).extend(usernames)
            if "cve-2025-31161" in output_lower or "auth bypass" in output_lower:
                discoveries["vulnerability"] = True
                discoveries.setdefault("cve", []).append("CVE-2025-31161")

        # Erlang cookie extraction
        cookie_match = re.search(r'erlang.*cookie.*?:\s*(\S+)', output, re.IGNORECASE)
        if cookie_match:
            discoveries["credential"] = True

        # Erlang OTP RCE
        if "erlang/otp" in output_lower and "uid=0(root)" in output:
            discoveries["root_shell"] = True
            discoveries["shell"] = True

        # tshark/PCAP FTP credentials (tab-separated USER/PASS)
        ftp_users = re.findall(r'USER\s+(\S+)', output)
        ftp_passes = re.findall(r'PASS\s+(\S+)', output)
        if ftp_users and ftp_passes:
            for u in ftp_users:
                if u.lower() not in ("anonymous", "ftp", "guest"):
                    discoveries["credential"] = True
                    break

        # vhost discovery
        vhost_matches = re.findall(r'Found:\s+(\S+\.(?:htb|local|internal))\s', output)
        if vhost_matches:
            discoveries.setdefault("web_paths", []).extend(vhost_matches)

        # Generic credential pattern (Phase 19 enhanced)
        for match in re.finditer(r'credential:\s*(\S+):(\S+)', output):
            discoveries["credential"] = True

        # Flag patterns
        flag_match = re.search(r'FLAG\{[^}]+\}', output)
        if flag_match:
            discoveries["flag"] = True

        # CVE patterns (universal)
        for cve_match in re.finditer(r'(CVE-\d{4}-\d+)', output):
            discoveries["vulnerability"] = True
            discoveries.setdefault("cve", []).append(cve_match.group(1))

        return discoveries

    def _parse_fullparse(
        self,
        command: str,
        output: str,
        agent_name: str,
    ) -> List[DiscoveryEvent]:
        """Phase 22: GPT-only intelligent fullparse.
        
        GPT-5.2-codex is the PRIMARY and ONLY interpreter. No regex pre-stages,
        no Venice. The LLM reads the raw output, reasons about what it contains,
        teaches the agent WHY certain patterns indicate discoveries, and
        extracts structured findings with full reasoning chains.
        
        This maximises interpretation quality at the cost of one GPT call per step.
        Regex fallback only fires if GPT is exhausted or offline.
        """
        flat_discoveries = {}
        source_stage = "gpt_codex"

        # ── PRIMARY: GPT-5.2-codex LLM interpreter (full reasoning) ──
        # Goes straight to GPT for maximum intelligence. Teaches agents
        # how it parsed what and why via InterpretationLessons.
        if (
            self._llm_interpreter
            and self._llm_interpreter.can_interpret()
            and len(output.strip()) > 30
        ):
            lesson = self._llm_interpreter.interpret(
                command=command,
                output=output,
                agent_name=agent_name,
            )
            if lesson and lesson.discoveries:
                flat_discoveries = lesson.discoveries
                source_stage = "gpt_codex"
                self._stats["stage3_hits"] += 1
                self._stats["lessons_produced"] += 1
                self._last_lesson = lesson
                logger.debug(
                    f"[BROKER-GPT] gpt-5.2-codex | {agent_name} | "
                    f"found: {list(flat_discoveries.keys())} | "
                    f"{lesson.latency_ms:.0f}ms"
                )

        # ── FALLBACK: Regex only if GPT exhausted/offline ──
        if not flat_discoveries:
            # Stage 1+2: SmartOutputParser (regex + nano-LLM)
            if self._sop:
                flat_discoveries = self._sop.parse(
                    command=command,
                    output=output,
                    agent_name=agent_name,
                )
                if flat_discoveries:
                    source_stage = "regex_fallback"
                    self._stats["stage1_hits"] += 1

            # Phase 19: Ultra-smart regex extraction
            if not flat_discoveries:
                flat_discoveries = self._phase19_regex_extract(command, output)
                if flat_discoveries:
                    source_stage = "regex_p19_fallback"
                    self._stats["stage1_hits"] += 1

        # # ── Venice stages COMMENTED OUT — Phase 22 GPT-only mode ──
        # # Venice was adding 5-7s latency per call with 6000ms ping.
        # # All interpretation now handled by GPT-5.2-codex.
        # if (
        #     not flat_discoveries
        #     and self._enable_venice
        #     and self._venice_calls < self._max_venice_calls
        #     and len(output.strip()) > 30
        # ):
        #     venice_discoveries = self._venice_rationalise(command, output)
        #     ...

        # Convert to DiscoveryEvents
        events = DiscoveryEvent.from_flat_discoveries(
            discoveries=flat_discoveries,
            source_stage=source_stage,
            command=command,
            agent=agent_name,
        )
        self._stats["total_events"] += len(events)

        return events

    def _venice_rationalise(
        self,
        command: str,
        output: str,
    ) -> Dict[str, Any]:
        """
        Stage 3: Venice rationaliser.

        Uses GLM 4.7 Flash (or Codex Mini fallback) to validate/extract
        discoveries from ambiguous output.
        """
        if not self._venice:
            return {}

        truncated = output[:1200] if len(output) > 1200 else output
        prompt = (
            f"Extract penetration testing discoveries from this output.\n"
            f"Command: {command[:200]}\n"
            f"Output:\n{truncated}\n\n"
            f"Reply with comma-separated discoveries: PORT:num, SERVICE:name, "
            f"CRED:user:pass, SHELL:type, CVE:id, or NONE if nothing found."
        )

        try:
            # Venice reasoning layer has a .reason() or .query() method
            response = ""
            if hasattr(self._venice, "reason"):
                result = self._venice.reason(prompt)
                response = result if isinstance(result, str) else str(result.get("response", ""))
            elif hasattr(self._venice, "query"):
                response = self._venice.query(prompt)
            elif hasattr(self._venice, "call"):
                response = self._venice.call(prompt)

            if not response or "NONE" in response.upper():
                return {}

            return self._parse_venice_response(response)

        except Exception as e:
            logger.debug(f"[BROKER-VENICE] Error: {e}")
            # Fallback: try codex mini through GPTManager
            if self._gpt:
                try:
                    response = self._gpt.gpt_request(
                        prompt=prompt,
                        task_type="classification",
                        agent_id="parser_venice_fallback",
                        max_tokens=150,
                    )
                    if response and "NONE" not in response.upper():
                        return self._parse_venice_response(response)
                except Exception:
                    pass
            return {}

    def _parse_venice_response(self, response: str) -> Dict[str, Any]:
        """Parse Venice compact discovery format."""
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
                shell_type = item.split(":", 1)[1].strip().lower()
                discoveries["shell"] = True
                if "root" in shell_type or "admin" in shell_type:
                    discoveries["root_shell"] = True
            elif item.startswith("CVE:"):
                cve_id = item.split(":", 1)[1].strip().upper()
                if cve_id.startswith("CVE-"):
                    discoveries.setdefault("cve", []).append(cve_id)
                    discoveries["vulnerability"] = True

        return discoveries

    def _gpt_finalise(
        self,
        command: str,
        output: str,
        agent_name: str,
    ) -> Dict[str, Any]:
        """Stage 4: GPT finaliser for edge cases."""
        if not self._gpt:
            return {}

        # Delegate to SOP's LLM path directly
        if self._sop and hasattr(self._sop, "_llm_parse"):
            result = self._sop._llm_parse(command, output, agent_name)
            return result or {}

        return {}

    # ── Phase 11.3: Agent Learning Interface ────────────────────────

    def get_last_lesson(self) -> Optional[Any]:
        """Get the last InterpretationLesson produced (for agent feedback)."""
        return self._last_lesson

    def get_learned_context(self, agent_name: str) -> str:
        """
        Get the accumulated output interpretation lessons for an agent.
        Used to inject into mentor/reasoning prompts so the agent learns
        to interpret output without LLM calls over time.
        """
        if self._llm_interpreter:
            return self._llm_interpreter.get_all_learned_context(agent_name)
        return ""

    def get_learned_patterns(self, agent_name: str) -> List[str]:
        """Get raw pattern list this agent has learned."""
        if self._llm_interpreter:
            return self._llm_interpreter.get_learned_patterns(agent_name)
        return []

    def get_stats(self) -> Dict[str, Any]:
        """Get broker statistics."""
        stats = dict(self._stats)
        # Phase 11.3: Include LLM interpreter stats
        if self._llm_interpreter:
            stats["llm_interpreter"] = self._llm_interpreter.get_stats()
        return stats

    def get_stage_distribution(self) -> Dict[str, float]:
        """Get percentage distribution across stages."""
        total = max(self._stats["total_calls"], 1)
        return {
            "regex_pct": round(self._stats["stage1_hits"] / total * 100, 1),
            "sop_llm_pct": round(self._stats["stage2_hits"] / total * 100, 1),
            "llm_interp_pct": round(self._stats["stage3_hits"] / total * 100, 1),
            "gpt_pct": round(self._stats["stage4_hits"] / total * 100, 1),
            "empty_pct": round(self._stats["empty_outputs"] / total * 100, 1),
        }
