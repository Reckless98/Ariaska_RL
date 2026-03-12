"""
Venice Reasoning Layer — Humanlike output analysis and discovery extraction.

Uses Venice AI's qwen3-coder-480b-a35b-instruct model to:
1. Parse command outputs for hidden discoveries (ports, creds, vulns, paths)
2. Generate "aha" insights — correlating current findings with prior knowledge
3. Accelerate learning via cross-episode pattern recognition
4. Provide humanlike reasoning narratives for agent decision context

Architecture:
    SmartOrchestrator._run_step()
        → command executes (simulated or live)
        → VeniceReasoningLayer.analyze_output(command, output, context)
            → Venice API call with structured prompt
            → Returns VeniceInsight with discoveries, aha_moments, suggested_next
        → Insights feed back into AttackContext for next decision
        → "aha" moments stored in cross-episode memory for accelerated learning

Cost: Venice qwen3-coder-480b-a35b-instruct
      Roughly 5× cheaper than codex-mini for equivalent reasoning quality.
"""

import logging
import time
import json
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from core.gpt_manager import GPTManager

logger = logging.getLogger("ariaska.venice_reasoning")


@dataclass
class AhaMoment:
    """
    An 'aha' insight — a humanlike discovery correlation or pattern recognition.
    
    Examples:
        - "Port 1524 is open AND it's ingreslock — this is a backdoor, instant root!"
        - "vsftpd 2.3.4 has a known backdoor triggered by :) in username"
        - "msfadmin:msfadmin worked on SSH — same creds might work on Telnet and MySQL"
        - "NFS is world-readable — can mount / and plant SSH authorized_keys for persistent access"
    """
    insight: str               # The humanlike reasoning text
    confidence: float          # 0.0–1.0 confidence in this insight
    discovery_type: str        # "correlation", "vulnerability", "credential_reuse", "chain", "pattern"
    related_services: List[str] = field(default_factory=list)
    action_suggestion: str = ""  # Suggested next command based on this insight
    priority: int = 5           # 1 (critical) to 10 (informational)
    episode_persistent: bool = True  # Whether this insight carries across episodes


@dataclass
class VeniceInsight:
    """
    Structured result from Venice reasoning layer analysis.
    
    Contains extracted discoveries, aha moments, and suggested next actions.
    """
    # Extracted discoveries from output
    discovered_ports: List[int] = field(default_factory=list)
    discovered_services: List[str] = field(default_factory=list)
    discovered_credentials: List[str] = field(default_factory=list)
    discovered_vulns: List[str] = field(default_factory=list)
    discovered_paths: List[str] = field(default_factory=list)
    discovered_users: List[str] = field(default_factory=list)
    
    # Humanlike reasoning
    aha_moments: List[AhaMoment] = field(default_factory=list)
    reasoning_narrative: str = ""  # Full reasoning chain in natural language
    
    # Actionable intelligence
    suggested_next_commands: List[str] = field(default_factory=list)
    exploitation_paths: List[str] = field(default_factory=list)
    risk_assessment: str = "low"  # "low", "medium", "high"
    
    # Meta
    tokens_used: int = 0
    latency_ms: int = 0
    model_used: str = "qwen3-coder-480b-a35b-instruct"
    
    @property
    def has_discoveries(self) -> bool:
        """Check if any new discoveries were found."""
        return bool(
            self.discovered_ports or self.discovered_services or
            self.discovered_credentials or self.discovered_vulns or
            self.discovered_paths or self.discovered_users
        )
    
    @property
    def has_aha(self) -> bool:
        """Check if any aha moments were generated."""
        return bool(self.aha_moments)
    
    @property
    def top_aha(self) -> Optional[AhaMoment]:
        """Get highest-priority aha moment."""
        if not self.aha_moments:
            return None
        return min(self.aha_moments, key=lambda a: a.priority)


class VeniceReasoningLayer:
    """
    Venice AI reasoning layer for humanlike output analysis and insight generation.
    
    This layer sits between command execution and state update, analyzing raw
    command output through Venice's qwen3-coder-480b-a35b-instruct model to:
    
    1. Extract discoveries that regex parsers might miss
    2. Correlate findings across the engagement (aha moments)
    3. Suggest optimal next actions based on holistic understanding
    4. Build a narrative reasoning chain for agent context enrichment
    
    The reasoning layer is ADDITIVE — it enhances the existing regex-based
    discovery parser, not replaces it. Both run in parallel.
    """
    
    def __init__(
        self,
        gpt_manager: "GPTManager",
        call_budget_per_episode: int = 30,  # Phase 9.1: doubled for deeper output analysis
        min_output_length: int = 30,
        enable_cross_episode_memory: bool = True,
        budget_controller: Optional[Any] = None,
    ):
        """
        Initialize the Venice reasoning layer.
        
        Args:
            gpt_manager: Shared GPTManager instance (has Venice client)
            call_budget_per_episode: Max Venice calls per episode (cost control)
            min_output_length: Minimum output length to trigger analysis
            enable_cross_episode_memory: Whether aha moments persist across episodes
            budget_controller: Optional AdaptiveBudgetController for unified budget tracking
        """
        self.gpt_manager = gpt_manager
        self.call_budget = call_budget_per_episode
        self._calls_this_episode = 0
        self.min_output_length = min_output_length
        self.enable_cross_episode_memory = enable_cross_episode_memory
        self._budget_controller = budget_controller
        
        # Cross-episode aha memory (persists across episodes for accelerated learning)
        self._aha_memory: List[AhaMoment] = []
        self._max_aha_memory = 50  # Cap to prevent unbounded growth
        
        # Episode-level tracking
        self._episode_discoveries: Set[str] = set()
        self._episode_insights: List[VeniceInsight] = []
        
        # Performance stats
        self.stats = {
            "total_calls": 0,
            "total_aha_moments": 0,
            "total_discoveries_found": 0,
            "total_tokens": 0,
            "avg_latency_ms": 0.0,
            "episodes_analyzed": 0,
        }
        
        logger.info(
            f"VeniceReasoningLayer initialized | Budget: {call_budget_per_episode}/ep | "
            f"Venice available: {gpt_manager.has_venice() if gpt_manager else False}"
        )
    
    def reset_episode(self) -> None:
        """Reset per-episode state (call at episode start)."""
        self._calls_this_episode = 0
        self._episode_discoveries.clear()
        self._episode_insights.clear()
        self.stats["episodes_analyzed"] += 1
    
    def can_analyze(self) -> bool:
        """Check if we have budget for another Venice analysis call.
        
        Phase 12.1: Also consults AdaptiveBudgetController when available
        to ensure Venice calls respect unified budget pressure.
        """
        if not self.gpt_manager or not self.gpt_manager.has_venice():
            return False
        if self._calls_this_episode >= self.call_budget:
            return False
        # Phase 12.1: Check unified budget controller
        if self._budget_controller is not None:
            try:
                if not self._budget_controller.can_call_venice():
                    return False
            except Exception:
                pass
        return True
    
    def _build_analysis_prompt(
        self,
        command: str,
        output: str,
        phase: str,
        known_discoveries: Dict[str, Any],
        prior_aha: List[str],
    ) -> str:
        """Build the structured prompt for Venice output analysis."""
        # Truncate very long outputs to save tokens
        max_output_chars = 2000
        if len(output) > max_output_chars:
            output = output[:max_output_chars] + f"\n... (truncated, {len(output)} total chars)"
        
        # Format known discoveries
        known_str = ""
        if known_discoveries:
            known_parts = []
            for key, val in known_discoveries.items():
                if val and isinstance(val, (set, list)):
                    known_parts.append(f"  {key}: {', '.join(str(v) for v in list(val)[:10])}")
                elif val and isinstance(val, str):
                    known_parts.append(f"  {key}: {val}")
            known_str = "\n".join(known_parts)
        
        # Format prior aha moments for continuity
        prior_str = ""
        if prior_aha:
            prior_str = "\n".join(f"  • {a}" for a in prior_aha[-5:])
        
        return f"""You are an expert penetration tester analyzing command output from a live engagement.

COMMAND EXECUTED: {command}
CURRENT PHASE: {phase}

=== COMMAND OUTPUT ===
{output}

=== KNOWN DISCOVERIES SO FAR ===
{known_str if known_str else "  (none yet)"}

=== PRIOR INSIGHTS ===
{prior_str if prior_str else "  (none yet)"}

ANALYZE this output and respond in JSON:
{{
    "discoveries": {{
        "ports": [list of newly discovered port numbers],
        "services": ["service:port pairs found"],
        "credentials": ["user:password pairs or hashes found"],
        "vulnerabilities": ["CVE IDs or vulnerability descriptions"],
        "paths": ["web paths, file paths, or network paths found"],
        "users": ["usernames discovered"]
    }},
    "aha_moments": [
        {{
            "insight": "A humanlike realization connecting findings",
            "type": "correlation|vulnerability|credential_reuse|chain|pattern",
            "confidence": 0.0-1.0,
            "priority": 1-10,
            "action": "suggested next command based on this insight"
        }}
    ],
    "reasoning": "Brief narrative of what this output tells us and why it matters",
    "suggested_next": ["top 3 suggested next commands"],
    "risk": "low|medium|high"
}}

CRITICAL RULES:
- Only report NEW discoveries not already in "KNOWN DISCOVERIES"
- Aha moments should be genuine insights, not obvious restatements
- Think like a human pentester: "Wait, if X is true AND Y is true, then..."
- Correlate across prior insights for deeper understanding
- Be concise but insightful"""
    
    def analyze_output(
        self,
        command: str,
        output: str,
        phase: str = "RECON",
        known_discoveries: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> VeniceInsight:
        """
        Analyze command output through Venice reasoning layer.
        
        Args:
            command: The command that was executed
            output: The raw command output
            phase: Current attack phase
            known_discoveries: Already-known discoveries (to avoid duplicates)
            force: Force analysis even if budget exceeded
            
        Returns:
            VeniceInsight with extracted discoveries and aha moments
        """
        # Budget/availability check
        if not force and not self.can_analyze():
            return VeniceInsight(reasoning_narrative="[Venice budget exceeded or unavailable]")
        
        # Skip trivial outputs
        if not output or len(output.strip()) < self.min_output_length:
            return VeniceInsight(reasoning_narrative="[Output too short for analysis]")
        
        # Skip [SIM] marker-only outputs
        if output.strip().startswith("[SIM]") and len(output.strip()) < 50:
            return VeniceInsight(reasoning_narrative="[Simulation marker only]")
        
        self._calls_this_episode += 1
        self.stats["total_calls"] += 1
        
        # Build prior aha context for continuity
        prior_aha = [a.insight for a in self._aha_memory[-5:]]
        
        # Build prompt
        prompt = self._build_analysis_prompt(
            command=command,
            output=output,
            phase=phase,
            known_discoveries=known_discoveries or {},
            prior_aha=prior_aha,
        )
        
        start_time = time.time()
        insight = VeniceInsight()
        
        try:
            # Call Venice through GPTManager's venice_client
            venice_client = self.gpt_manager.venice_client
            venice_model = self.gpt_manager.venice_model
            
            response = venice_client.chat.completions.create(
                model=venice_model,
                messages=[
                    {"role": "system", "content": "You are an expert penetration tester. Analyze command output and provide structured insights in JSON format only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=750,  # Phase 11.5: +50% (was 500) for deeper reasoning analysis
                temperature=0.4,  # Lower temp for more precise analysis
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            insight.latency_ms = latency_ms
            insight.model_used = venice_model
            
            # Track tokens
            if hasattr(response, 'usage') and response.usage:
                insight.tokens_used = response.usage.total_tokens
                self.stats["total_tokens"] += insight.tokens_used
                # Update GPTManager Venice stats
                self.gpt_manager.stats_venice["total_requests"] += 1
                self.gpt_manager.stats_venice["successes"] += 1
                self.gpt_manager.stats_venice["tokens_used"] += insight.tokens_used
                # Phase 12.1: Unified budget accounting — Venice tokens count
                # against the global token budget to prevent invisible spend
                if hasattr(self.gpt_manager, 'tokens_used'):
                    self.gpt_manager.tokens_used += insight.tokens_used
                # Phase 12.1: Record in AdaptiveBudgetController
                if self._budget_controller is not None:
                    try:
                        self._budget_controller.record_venice_call(
                            tokens_used=insight.tokens_used
                        )
                    except Exception:
                        pass
            
            # Parse response
            content = ""
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content or ""
            
            if content:
                insight = self._parse_venice_response(content, insight)
                
                # Store aha moments in cross-episode memory
                if insight.has_aha and self.enable_cross_episode_memory:
                    for aha in insight.aha_moments:
                        if aha.episode_persistent:
                            self._aha_memory.append(aha)
                            self.stats["total_aha_moments"] += 1
                    # Cap memory
                    if len(self._aha_memory) > self._max_aha_memory:
                        self._aha_memory = self._aha_memory[-self._max_aha_memory:]
                
                if insight.has_discoveries:
                    disc_count = (
                        len(insight.discovered_ports) + len(insight.discovered_services) +
                        len(insight.discovered_credentials) + len(insight.discovered_vulns)
                    )
                    self.stats["total_discoveries_found"] += disc_count
                
                # Update running average latency
                n = self.stats["total_calls"]
                avg = self.stats["avg_latency_ms"]
                self.stats["avg_latency_ms"] = avg + (latency_ms - avg) / n
            
            self._episode_insights.append(insight)
            
            # Log insight summary
            if insight.has_aha:
                for aha in insight.aha_moments:
                    logger.info(f"💡 AHA: {aha.insight[:80]}... (conf={aha.confidence:.1f}, pri={aha.priority})")
            
            return insight
            
        except Exception as e:
            logger.warning(f"Venice reasoning analysis failed: {e}")
            self.gpt_manager.stats_venice["failures"] = self.gpt_manager.stats_venice.get("failures", 0) + 1
            
            # Phase 11.5: Fallback to local-llm when Venice fails
            try:
                logger.info(f"[VENICE→CODEX FALLBACK] Attempting local-llm for reasoning analysis")
                fallback_response = self.gpt_manager.gpt_request(
                    prompt=prompt,
                    task_type="reasoning",
                    agent_id="venice_fallback",
                    max_tokens=750,  # Phase 11.5: generous budget for deep reasoning
                    model="local-llm",
                )
                if fallback_response:
                    latency_ms = int((time.time() - start_time) * 1000)
                    insight.latency_ms = latency_ms
                    insight.model_used = "local-llm"
                    insight = self._parse_venice_response(fallback_response, insight)
                    insight.reasoning_narrative = (
                        f"[local-llm fallback] {insight.reasoning_narrative}"
                    )
                    logger.info(f"[VENICE→CODEX FALLBACK] Success — discoveries found: {insight.has_discoveries}")
                    self._episode_insights.append(insight)
                    return insight
            except Exception as fallback_err:
                logger.debug(f"[VENICE→CODEX FALLBACK] Also failed: {fallback_err}")
            
            insight.reasoning_narrative = f"[Venice analysis error: {str(e)[:50]}]"
            return insight
    
    def _parse_venice_response(self, content: str, insight: VeniceInsight) -> VeniceInsight:
        """Parse Venice JSON response into VeniceInsight."""
        try:
            # Try to extract JSON from response
            # Venice sometimes wraps in markdown code blocks
            json_str = content.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            data = json.loads(json_str)
            
            # Extract discoveries
            disc = data.get("discoveries", {})
            if isinstance(disc, dict):
                insight.discovered_ports = [int(p) for p in disc.get("ports", []) if str(p).isdigit()]
                insight.discovered_services = disc.get("services", [])
                insight.discovered_credentials = disc.get("credentials", [])
                insight.discovered_vulns = disc.get("vulnerabilities", [])
                insight.discovered_paths = disc.get("paths", [])
                insight.discovered_users = disc.get("users", [])
            
            # Extract aha moments
            aha_list = data.get("aha_moments", [])
            for aha_data in aha_list:
                if isinstance(aha_data, dict) and aha_data.get("insight"):
                    aha = AhaMoment(
                        insight=aha_data["insight"],
                        confidence=float(aha_data.get("confidence", 0.5)),
                        discovery_type=aha_data.get("type", "pattern"),
                        action_suggestion=aha_data.get("action", ""),
                        priority=int(aha_data.get("priority", 5)),
                    )
                    insight.aha_moments.append(aha)
            
            # Extract other fields
            insight.reasoning_narrative = data.get("reasoning", "")
            insight.suggested_next_commands = data.get("suggested_next", [])
            insight.risk_assessment = data.get("risk", "low")
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug(f"Venice response parse failed (using raw): {e}")
            # Fallback: use raw content as reasoning narrative
            insight.reasoning_narrative = content[:500]
        
        return insight
    
    def get_aha_context_for_agent(self, agent_name: str = "", limit: int = 5) -> str:
        """
        Get accumulated aha insights formatted for agent decision context.
        
        Returns a text block that can be injected into AttackContext or mentor prompts
        to give agents accumulated situational awareness.
        """
        if not self._aha_memory:
            return ""
        
        # Get most relevant recent aha moments
        recent = sorted(self._aha_memory[-limit:], key=lambda a: a.priority)
        
        lines = ["=== ACCUMULATED INSIGHTS (Venice Reasoning) ==="]
        for aha in recent:
            lines.append(f"  💡 [{aha.discovery_type}] {aha.insight}")
            if aha.action_suggestion:
                lines.append(f"     → Suggested: {aha.action_suggestion}")
        
        return "\n".join(lines)
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of Venice reasoning for this episode."""
        return {
            "calls_made": self._calls_this_episode,
            "budget": self.call_budget,
            "insights_generated": len(self._episode_insights),
            "aha_moments": sum(len(i.aha_moments) for i in self._episode_insights),
            "discoveries_found": sum(1 for i in self._episode_insights if i.has_discoveries),
            "total_tokens": sum(i.tokens_used for i in self._episode_insights),
            "aha_memory_size": len(self._aha_memory),
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cumulative Venice reasoning stats."""
        return dict(self.stats)
