#!/usr/bin/env python3
"""
core/execution/smart_output_parser.py — ARIASKA Smart Output Parser v1.0

Two-stage output parsing wrapping the existing OutputParser:
  1. Fast regex pass via OutputParser (zero cost) — handles 80%+ of outputs
  2. Nano-LLM fallback (gpt-5-nano) — for unparseable or ambiguous output

The nano-LLM path is ultra-cheap (~$0.0001/call) and only triggered
when the regex parser finds nothing meaningful in non-trivial output.

Usage:
    parser = SmartOutputParser(gpt_manager=gpt)
    discoveries = parser.parse(command="nmap -sV 10.10.10.10", output=raw_output)
"""

import re
import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.gpt_manager import GPTManager

logger = logging.getLogger("ariaska.smart_output_parser")

# Nano-LLM model for output parsing (ultra-cheap)
PARSER_MODEL = "gpt-5-nano"

# Minimum meaningful output length to consider LLM parsing
MIN_OUTPUT_FOR_LLM = 30


class SmartOutputParser:
    """
    Two-stage output parser: existing OutputParser regex + nano-LLM fallback.
    
    Designed for both SIM and LIVE outputs. The regex stage handles
    standard tool output formats (nmap, hydra, etc.). The LLM stage
    handles non-standard or mixed outputs.
    """

    def __init__(
        self,
        gpt_manager: Optional["GPTManager"] = None,
        enable_llm: bool = True,
        max_llm_calls_per_episode: int = 20,
    ):
        self._gpt = gpt_manager
        self._enable_llm = enable_llm and gpt_manager is not None
        self._max_llm_calls = max_llm_calls_per_episode
        self._llm_calls_this_episode: int = 0
        
        # Wrap existing OutputParser for regex stage
        self._regex_parser: Optional[Any] = None
        try:
            from core.execution.output_parser import OutputParser
            self._regex_parser = OutputParser()
        except ImportError:
            logger.warning("OutputParser not available, LLM-only mode")
        
        # Stats
        self._stats = {
            "regex_hits": 0,
            "llm_calls": 0,
            "llm_discoveries": 0,
            "empty_outputs": 0,
            "total_calls": 0,
        }

    def reset_episode(self) -> None:
        """Reset per-episode counters."""
        self._llm_calls_this_episode = 0

    def parse(
        self,
        command: str,
        output: str,
        agent_name: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Parse command output for discoveries using two-stage pipeline.
        
        Returns dict of discoveries compatible with SmartOrchestrator:
        {"open_port": [80], "service": ["http"], "credential": True, ...}
        """
        self._stats["total_calls"] += 1
        
        if not output or len(output.strip()) < 5:
            self._stats["empty_outputs"] += 1
            return {}
        
        # ── Stage 1: Regex via existing OutputParser ────────────────
        regex_discoveries = {}
        if self._regex_parser:
            try:
                parsed = self._regex_parser.parse(command, output)
                regex_discoveries = self._parsed_output_to_discoveries(parsed)
            except Exception as e:
                logger.debug(f"[SMART-PARSER] Regex parse error: {e}")
        
        if regex_discoveries:
            self._stats["regex_hits"] += 1
            return regex_discoveries
        
        # ── Stage 2: LLM fallback (nano model) ─────────────────────
        if (
            self._enable_llm
            and self._llm_calls_this_episode < self._max_llm_calls
            and len(output.strip()) > MIN_OUTPUT_FOR_LLM
            and not self._is_trivial_output(output)
        ):
            llm_result = self._llm_parse(command, output, agent_name)
            if llm_result:
                self._stats["llm_calls"] += 1
                self._stats["llm_discoveries"] += 1
                self._llm_calls_this_episode += 1
                return llm_result
        
        return regex_discoveries

    # ------------------------------------------------------------------
    # Convert ParsedOutput to flat discovery dict
    # ------------------------------------------------------------------

    @staticmethod
    def _parsed_output_to_discoveries(parsed: Any) -> Dict[str, Any]:
        """Convert ParsedOutput dataclass to flat discovery dict."""
        discoveries: Dict[str, Any] = {}
        
        if hasattr(parsed, "open_ports") and parsed.open_ports:
            discoveries["open_port"] = list(parsed.open_ports)
        
        if hasattr(parsed, "services") and parsed.services:
            discoveries["service"] = list(set(parsed.services.values()))
        
        if hasattr(parsed, "credentials") and parsed.credentials:
            discoveries["credential"] = True
        
        if hasattr(parsed, "users") and parsed.users:
            discoveries["user"] = list(parsed.users)
        
        if hasattr(parsed, "vulnerabilities") and parsed.vulnerabilities:
            discoveries["vulnerability"] = True
            cves = [v for v in parsed.vulnerabilities if v.startswith("CVE-")]
            if cves:
                discoveries["cve"] = cves
        
        if hasattr(parsed, "web_paths") and parsed.web_paths:
            discoveries["web_path"] = list(parsed.web_paths)
        
        if hasattr(parsed, "shares") and parsed.shares:
            discoveries["smb_share"] = list(parsed.shares)
        
        if hasattr(parsed, "sessions") and parsed.sessions:
            discoveries["shell"] = True
            for s in parsed.sessions:
                if s.get("type") in ("root", "system", "admin"):
                    discoveries["root_shell"] = True
        
        if hasattr(parsed, "os_info") and parsed.os_info:
            discoveries["version_info"] = [parsed.os_info]
        
        return discoveries

    # ------------------------------------------------------------------
    # LLM-based parsing (nano model)
    # ------------------------------------------------------------------

    def _llm_parse(
        self,
        command: str,
        output: str,
        agent_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Use gpt-5-nano to parse output that regex couldn't handle.
        """
        if not self._gpt:
            return None
        
        # Truncate output to keep costs minimal
        truncated = output[:1500] if len(output) > 1500 else output
        
        prompt = f"""Parse this penetration testing tool output and extract discoveries as JSON.

Command: {command}
Output:
```
{truncated}
```

Return ONLY a JSON object with these possible keys (include only what you find):
- "open_port": [list of integer port numbers]
- "service": [list of service name strings]  
- "version_info": [list of "service version" strings]
- "credential": true if credentials found
- "user": [list of usernames]
- "vulnerability": true if vulnerabilities found
- "cve": [list of CVE IDs]
- "shell": true if shell access obtained
- "root_shell": true if root/admin shell
- "web_path": [list of web paths found]

If nothing useful found, return {{}}.
JSON:"""

        try:
            response = self._gpt.gpt_request(
                prompt=prompt,
                task_type="classification",
                agent_id=f"parser_{agent_name}",
                max_tokens=200,
                model=PARSER_MODEL,
            )
            
            if not response:
                return None
            
            # Extract JSON from response
            json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                # Filter out empty values
                result = {k: v for k, v in result.items() if v}
                if result:
                    logger.debug(f"[SMART-PARSER-LLM] Found: {list(result.keys())}")
                    return result
            
        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"[SMART-PARSER-LLM] Parse failed: {e}")
        
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_trivial_output(output: str) -> bool:
        """Check if output is trivial (all errors, SIM markers, etc.)."""
        stripped = output.strip()
        if stripped.startswith("[SIM]") and len(stripped) < 50:
            return True
        lines = stripped.split("\n")
        if len(lines) <= 1:
            return True
        error_markers = ["error", "failed", "denied", "refused", "timeout", "not found"]
        error_count = sum(
            1 for line in lines
            if any(e in line.lower() for e in error_markers)
        )
        if len(lines) > 2 and error_count / len(lines) > 0.8:
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get parser statistics."""
        return dict(self._stats)
