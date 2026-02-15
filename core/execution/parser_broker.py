#!/usr/bin/env python3
"""
core/execution/parser_broker.py — Phase 10.0: 4-Stage Parser Broker

Unified discovery parsing pipeline:
  Stage 1: Regex (OutputParser)     — free, handles 80%+
  Stage 2: SmartOutputParser (SOP)  — existing nano-LLM fallback
  Stage 3: Venice rationaliser      — GLM 4.7 Flash (or Codex Mini fallback)
  Stage 4: GPT finaliser            — gpt-5-nano for edge cases

Each stage only fires if prior stages found nothing meaningful.
All stages emit DiscoveryEvent objects.

The Venice rationaliser validates and enriches discoveries, not extract.
It receives Stage 1/2 output + command context and says:
  "Given this output, are the extracted discoveries correct? What's missing?"

Author: Filip Volf / Ariaska System
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from core.execution.discovery_event import DiscoveryEvent, DiscoveryType

if TYPE_CHECKING:
    from core.gpt_manager import GPTManager

logger = logging.getLogger("ariaska.parser_broker")


class ParserBroker:
    """
    4-stage output parsing pipeline.

    Usage:
        broker = ParserBroker(gpt_manager=gpt, venice=venice_layer)
        events = broker.parse(command, output, agent_name)
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
    ):
        self._gpt = gpt_manager
        self._venice = venice
        self._enable_venice = enable_venice and venice is not None
        self._enable_gpt = enable_gpt and gpt_manager is not None
        self._max_llm_calls = max_llm_calls_per_episode
        self._max_venice_calls = max_venice_calls_per_episode
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

        # Stats
        self._stats = {
            "total_calls": 0,
            "stage1_hits": 0,    # regex
            "stage2_hits": 0,    # SOP LLM
            "stage3_hits": 0,    # Venice
            "stage4_hits": 0,    # GPT finaliser
            "empty_outputs": 0,
            "total_events": 0,
        }

    def reset_episode(self) -> None:
        """Reset per-episode counters."""
        self._llm_calls = 0
        self._venice_calls = 0
        if self._sop:
            self._sop.reset_episode()

    def parse(
        self,
        command: str,
        output: str,
        agent_name: str = "unknown",
    ) -> List[DiscoveryEvent]:
        """
        Run the 4-stage pipeline and return DiscoveryEvents.

        Args:
            command: The command that was executed
            output: Raw command output text
            agent_name: Which agent ran this

        Returns:
            List of DiscoveryEvent objects
        """
        self._stats["total_calls"] += 1

        if not output or len(output.strip()) < 5:
            self._stats["empty_outputs"] += 1
            return []

        # ── Stage 1+2: SmartOutputParser (regex + nano-LLM) ───────
        flat_discoveries = {}
        source_stage = "regex"

        if self._sop:
            flat_discoveries = self._sop.parse(
                command=command,
                output=output,
                agent_name=agent_name,
            )
            if flat_discoveries:
                # Determine which stage produced the result
                sop_stats = self._sop.get_stats()
                if sop_stats.get("llm_discoveries", 0) > 0:
                    source_stage = "sop_llm"
                    self._stats["stage2_hits"] += 1
                else:
                    source_stage = "regex"
                    self._stats["stage1_hits"] += 1

        # ── Stage 3: Venice rationaliser ──────────────────────────
        if (
            not flat_discoveries
            and self._enable_venice
            and self._venice_calls < self._max_venice_calls
            and len(output.strip()) > 30
        ):
            venice_discoveries = self._venice_rationalise(command, output)
            if venice_discoveries:
                flat_discoveries = venice_discoveries
                source_stage = "venice"
                self._stats["stage3_hits"] += 1
                self._venice_calls += 1

        # ── Stage 4: GPT finaliser (last resort) ─────────────────
        if (
            not flat_discoveries
            and self._enable_gpt
            and self._llm_calls < self._max_llm_calls
            and len(output.strip()) > 50
        ):
            gpt_discoveries = self._gpt_finalise(command, output, agent_name)
            if gpt_discoveries:
                flat_discoveries = gpt_discoveries
                source_stage = "gpt_finaliser"
                self._stats["stage4_hits"] += 1
                self._llm_calls += 1

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

    def get_stats(self) -> Dict[str, Any]:
        """Get broker statistics."""
        return dict(self._stats)

    def get_stage_distribution(self) -> Dict[str, float]:
        """Get percentage distribution across stages."""
        total = max(self._stats["total_calls"], 1)
        return {
            "regex_pct": round(self._stats["stage1_hits"] / total * 100, 1),
            "sop_llm_pct": round(self._stats["stage2_hits"] / total * 100, 1),
            "venice_pct": round(self._stats["stage3_hits"] / total * 100, 1),
            "gpt_pct": round(self._stats["stage4_hits"] / total * 100, 1),
            "empty_pct": round(self._stats["empty_outputs"] / total * 100, 1),
        }
