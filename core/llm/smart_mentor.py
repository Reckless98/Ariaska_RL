"""
Smart Mentor - Structured GPT prompting for intelligent command generation.

This module provides rich context to the LLM for generating high-quality
pentesting commands based on current attack state, phase, and history.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum

from ..commands.command_registry import (
    AttackPhase,
    CommandTemplate,
    CommandChoice,
    COMMAND_REGISTRY,
    get_valid_commands_for_state,
    get_phase_from_state,
    render_command,
    get_command_names_for_prompt
)
from ..commands.learned_commands import (
    LearnedCommandStore,
    get_learned_store
)


@dataclass
class AttackContext:
    """
    Rich context about the current attack state for LLM prompting.
    
    Attributes:
        target: Target IP or hostname
        current_phase: Current phase in attack chain
        state_flags: Dictionary of discovered facts
        command_history: Recent commands executed
        discoveries: Important findings (open ports, users, etc.)
        failed_attempts: Commands that didn't work
        time_budget: Remaining time or tokens
        difficulty: Target difficulty (easy, medium, hard, insane)
        platform: Target platform (linux, windows, unknown)
        services_found: Discovered services
    """
    target: str
    current_phase: AttackPhase = AttackPhase.RECON
    state_flags: Dict[str, bool] = field(default_factory=dict)
    command_history: List[str] = field(default_factory=list)
    discoveries: Dict[str, Any] = field(default_factory=dict)
    failed_attempts: List[str] = field(default_factory=list)
    time_budget: Optional[int] = None
    difficulty: str = "unknown"
    platform: str = "unknown"
    services_found: List[str] = field(default_factory=list)
    
    def to_narrative(self) -> str:
        """Convert context to human-readable narrative for LLM."""
        lines = []
        
        # Target info
        lines.append(f"TARGET: {self.target}")
        lines.append(f"PHASE: {self.current_phase.name}")
        lines.append(f"PLATFORM: {self.platform}")
        if self.difficulty != "unknown":
            lines.append(f"DIFFICULTY: {self.difficulty}")
        
        # Services
        if self.services_found:
            lines.append(f"\nDISCOVERED SERVICES:")
            for svc in self.services_found:
                lines.append(f"  • {svc}")
        
        # Key discoveries
        if self.discoveries:
            lines.append(f"\nKEY DISCOVERIES:")
            for key, value in self.discoveries.items():
                if isinstance(value, list):
                    lines.append(f"  • {key}: {', '.join(str(v) for v in value[:5])}")
                else:
                    lines.append(f"  • {key}: {value}")
        
        # Recent commands (last 5)
        if self.command_history:
            lines.append(f"\nRECENT COMMANDS (last 5):")
            for cmd in self.command_history[-5:]:
                lines.append(f"  > {cmd[:80]}...")
        
        # Failed attempts (to avoid)
        if self.failed_attempts:
            lines.append(f"\nFAILED ATTEMPTS (avoid these):")
            for cmd in self.failed_attempts[-3:]:
                lines.append(f"  ✗ {cmd[:60]}...")
        
        # Current state flags summary
        active_flags = [k for k, v in self.state_flags.items() if v]
        if active_flags:
            lines.append(f"\nCURRENT STATE:")
            for flag in active_flags[:10]:
                lines.append(f"  ✓ {flag}")
        
        return "\n".join(lines)
    
    def add_discovery(self, key: str, value: Any) -> None:
        """Add a discovery to the context."""
        if key in self.discoveries:
            if isinstance(self.discoveries[key], list):
                if isinstance(value, list):
                    self.discoveries[key].extend(value)
                else:
                    self.discoveries[key].append(value)
            else:
                self.discoveries[key] = [self.discoveries[key], value]
        else:
            self.discoveries[key] = value
    
    def set_state_flag(self, flag: str, value: bool = True) -> None:
        """Set a state flag."""
        self.state_flags[flag] = value
        
        # Auto-detect phase advancement
        self.current_phase = get_phase_from_state(self.state_flags)
    
    def add_service(self, service: str, port: Optional[int] = None) -> None:
        """Add a discovered service."""
        svc_str = f"{service}:{port}" if port else service
        if svc_str not in self.services_found:
            self.services_found.append(svc_str)
        
        # Set appropriate state flag
        service_lower = service.lower()
        if "http" in service_lower or "web" in service_lower:
            self.set_state_flag("http_service_found")
        elif "ssh" in service_lower:
            self.set_state_flag("ssh_service_found")
        elif "smb" in service_lower or "445" in str(port):
            self.set_state_flag("smb_service_found")
        elif "ftp" in service_lower:
            self.set_state_flag("ftp_service_found")
        elif "ldap" in service_lower:
            self.set_state_flag("ldap_service_found")
        elif "dns" in service_lower:
            self.set_state_flag("dns_service_found")
        elif "snmp" in service_lower:
            self.set_state_flag("snmp_service_found")
        elif "winrm" in service_lower or "5985" in str(port):
            self.set_state_flag("winrm_service_found")
        elif "kerberos" in service_lower or "88" in str(port):
            self.set_state_flag("kerberos_service_found")


@dataclass
class MentorResponse:
    """
    Structured response from the smart mentor.
    
    Attributes:
        command: The actual command to execute
        template_name: Name of the CommandTemplate used
        params: Parameters used
        reasoning: Why this command was chosen
        confidence: Confidence in this choice (0-1)
        next_phase_hint: Suggested next phase
        alternative_commands: Backup commands if this fails
        phase: Current attack phase
        tokens_used: Tokens consumed by this mentor call
    """
    command: str
    template_name: str
    params: Dict[str, str]
    reasoning: str
    confidence: float
    next_phase_hint: Optional[str] = None
    alternative_commands: List[str] = field(default_factory=list)
    phase: AttackPhase = AttackPhase.RECON
    tokens_used: int = 0
    
    @property
    def is_valid(self) -> bool:
        """Check if this is a valid response."""
        return bool(self.command and self.template_name)


class SmartMentor:
    """
    Intelligent mentor that generates pentesting commands using LLM.
    
    Provides structured prompts with rich context and validates
    output against the command registry.
    """
    
    def __init__(
        self,
        llm_client: Any,
        learned_store: Optional[LearnedCommandStore] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_retries: int = 2
    ):
        """
        Initialize the smart mentor.
        
        Args:
            llm_client: LLM client with chat completion capability
            learned_store: Store for learned commands
            model: Model to use
            temperature: Sampling temperature
            max_retries: Max retries for failed parses
        """
        self.llm_client = llm_client
        self.learned_store = learned_store or get_learned_store()
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the mentor."""
        return """You are an expert penetration tester and red team operator with deep knowledge of:
- Network reconnaissance and enumeration
- Web application security testing
- Active Directory attacks and lateral movement
- Linux and Windows privilege escalation
- Advanced techniques like port knocking, tunneling, and evasion

Your role is to select the BEST next command for the current attack phase.

CRITICAL RULES:
1. ALWAYS select from the provided command list - never invent commands
2. NEVER repeat a command from the "COMMANDS ALREADY TRIED" section!
3. If you see a command was tried multiple times, it's STUCK - pick something completely different
4. Consider the current phase and what discoveries have been made
5. Progress through phases: Recon → Enumeration → Exploitation → PrivEsc → Lateral → Post-Ex
6. Be creative but realistic - use the right tool for the situation
7. Variety is key - try different approaches if current ones aren't working

ANTI-LOOP BEHAVIOR:
- If nmap was already used, try a different scanner (masscan, rustscan)
- If one web tool failed, try another (gobuster → feroxbuster → dirb)
- If SSH failed, try SMB or other services
- NEVER select the same command twice in a row

OUTPUT FORMAT (JSON only):
{
    "selected_command": "template_name from the list",
    "parameters": {"param1": "value1", "param2": "value2"},
    "reasoning": "Brief explanation of why this command",
    "confidence": 0.8,
    "next_phase_hint": "what to try if this works"
}"""
    
    def _build_user_prompt(
        self,
        context: AttackContext,
        valid_commands: List[CommandTemplate]
    ) -> str:
        """Build the user prompt with context and available commands."""
        lines = []
        
        # Attack context narrative
        lines.append("=== CURRENT ATTACK STATE ===")
        lines.append(context.to_narrative())
        lines.append("")
        
        # === CRITICAL: Show what was ALREADY TRIED to prevent loops ===
        if context.command_history:
            lines.append("=== COMMANDS ALREADY TRIED (DO NOT REPEAT) ===")
            # Show last 15 commands with count of how many times each was used
            recent_commands = context.command_history[-15:]
            from collections import Counter
            cmd_counts = Counter(recent_commands)
            for cmd, count in cmd_counts.most_common(10):
                # Truncate long commands
                cmd_display = cmd[:60] + "..." if len(cmd) > 60 else cmd
                if count > 1:
                    lines.append(f"  ❌ {cmd_display} (tried {count}x - AVOID!)")
                else:
                    lines.append(f"  ⚠️ {cmd_display} (already tried)")
            lines.append("")
            lines.append("⚠️ IMPORTANT: Choose a DIFFERENT command from the list below!")
            lines.append("")
        
        # Available commands
        lines.append("=== AVAILABLE COMMANDS ===")
        lines.append(f"(Phase: {context.current_phase.name})")
        lines.append("")
        
        for cmd in valid_commands[:25]:  # Limit to avoid token overflow
            params = ", ".join(cmd.required_params) if cmd.required_params else "none"
            optional = ", ".join(f"{k}={v}" for k, v in list(cmd.optional_params.items())[:3])
            
            lines.append(f"• {cmd.name}")
            lines.append(f"  Required params: {params}")
            if optional:
                lines.append(f"  Optional: {optional}")
            lines.append(f"  Description: {cmd.description}")
            
            # Include WHY/WHEN context for smarter decisions
            if hasattr(cmd, 'why') and cmd.why:
                lines.append(f"  WHY: {cmd.why}")
            if hasattr(cmd, 'when') and cmd.when:
                lines.append(f"  WHEN: {cmd.when}")
            if hasattr(cmd, 'not_when') and cmd.not_when:
                lines.append(f"  AVOID: {cmd.not_when}")
            if hasattr(cmd, 'follows_after') and cmd.follows_after:
                lines.append(f"  FOLLOWS: {', '.join(cmd.follows_after[:3])}")
            if hasattr(cmd, 'enables') and cmd.enables:
                lines.append(f"  ENABLES: {', '.join(cmd.enables[:3])}")
            
            lines.append("")
        
        # Learned command suggestions (filtered by valid_commands if provided)
        learned_suggestions = self.learned_store.get_commands_for_prompt(
            phase=context.current_phase,
            preconditions=set(k for k, v in context.state_flags.items() if v),
            limit=5  # Get more, then filter
        )
        
        # Filter learned suggestions to only include commands in valid_commands
        if learned_suggestions and valid_commands:
            valid_names = {cmd.name.lower() for cmd in valid_commands}
            filtered_suggestions = []
            for suggestion in learned_suggestions:
                # Extract command name from suggestion string (format: "• command_name: ...")
                if "•" in suggestion:
                    cmd_name = suggestion.split("•")[1].split(":")[0].strip().lower()
                    if cmd_name in valid_names:
                        filtered_suggestions.append(suggestion)
            learned_suggestions = filtered_suggestions[:3]
        
        if learned_suggestions:
            lines.append("=== PROVEN SUCCESSFUL COMMANDS ===")
            for suggestion in learned_suggestions:
                lines.append(suggestion)
            lines.append("")
        
        # Instruction
        lines.append("=== YOUR TASK ===")
        lines.append("Select the BEST next command from the AVAILABLE COMMANDS list above.")
        lines.append("IMPORTANT: Only use commands from the list - do not invent or use other commands.")
        lines.append("Fill in all required parameters based on the target and context.")
        lines.append("Respond with ONLY valid JSON, no markdown or explanation outside JSON.")
        
        return "\n".join(lines)
    
    def _parse_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response into structured format."""
        # Try to extract JSON from response
        text = response_text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            # Find the end
            lines = text.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith("```") and not in_json:
                    in_json = True
                    continue
                elif line.startswith("```") and in_json:
                    break
                elif in_json:
                    json_lines.append(line)
            text = "\n".join(json_lines)
        
        # Try to find JSON object
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group()
        
        try:
            data = json.loads(text)
            return data
        except json.JSONDecodeError:
            return None
    
    def _validate_and_build_response(
        self,
        parsed: Dict[str, Any],
        context: AttackContext,
        valid_commands: List[CommandTemplate]
    ) -> Optional[MentorResponse]:
        """Validate parsed response and build MentorResponse."""
        template_name = parsed.get("selected_command", "")
        params = parsed.get("parameters", {})
        reasoning = parsed.get("reasoning", "")
        confidence = float(parsed.get("confidence", 0.5))
        next_hint = parsed.get("next_phase_hint", "")
        
        # Find the template
        template = COMMAND_REGISTRY.get(template_name)
        if not template:
            # Try fuzzy match
            for name, cmd in COMMAND_REGISTRY.items():
                if template_name.lower() in name.lower() or name.lower() in template_name.lower():
                    template = cmd
                    template_name = name
                    break
        
        if not template:
            return None
        
        # Ensure required params are filled
        for req_param in template.required_params:
            if req_param not in params or not params[req_param]:
                # Try to fill from context
                if req_param == "target":
                    params[req_param] = context.target
                elif req_param == "url" and context.target:
                    # Construct URL from target
                    port = 80
                    for svc in context.services_found:
                        if "http" in svc.lower():
                            parts = svc.split(":")
                            if len(parts) > 1:
                                try:
                                    port = int(parts[1])
                                except ValueError:
                                    pass
                    proto = "https" if port == 443 else "http"
                    params[req_param] = f"{proto}://{context.target}:{port}"
                elif req_param == "ports" and "open_ports" in context.discoveries:
                    params[req_param] = ",".join(str(p) for p in context.discoveries["open_ports"][:10])
                else:
                    # Still missing required param
                    return None
        
        # Render the command
        try:
            command = render_command(template, params)
        except ValueError as e:
            return None
        
        return MentorResponse(
            command=command,
            template_name=template_name,
            params=params,
            reasoning=reasoning,
            confidence=confidence,
            next_phase_hint=next_hint,
            phase=template.phase
        )
    
    async def get_command_async(
        self,
        context: AttackContext,
        filtered_commands: Optional[List[CommandTemplate]] = None
    ) -> MentorResponse:
        """
        Get the next command asynchronously.
        
        Args:
            context: Current attack context
            filtered_commands: Pre-filtered commands (by agent role). If None, uses global registry.
            
        Returns:
            MentorResponse with command and reasoning
        """
        # Use pre-filtered commands if provided (role-aware), else fallback to global
        if filtered_commands:
            valid_commands = filtered_commands
        else:
            # Get valid commands for current state
            valid_commands = get_valid_commands_for_state(
                context.state_flags,
                context.current_phase
            )
            
            # If no valid commands for current phase, try all phases
            if not valid_commands:
                valid_commands = get_valid_commands_for_state(context.state_flags)
            
            # If still none, use recon commands (always valid)
            if not valid_commands:
                valid_commands = [
                    cmd for cmd in COMMAND_REGISTRY.values()
                    if cmd.phase == AttackPhase.RECON and not cmd.preconditions
                ]
        
        # Build prompts
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(context, valid_commands)
        
        # Track tokens for this call (accumulate across retries)
        tokens_used = 0
        
        # Call LLM
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=500
                )
                
                # Capture token usage from API response (accumulate!)
                if hasattr(response, 'usage') and response.usage:
                    tokens_used += response.usage.total_tokens
                
                response_text = response.choices[0].message.content
                
                # Parse response
                parsed = self._parse_response(response_text)
                if not parsed:
                    continue
                
                # Validate and build response
                mentor_response = self._validate_and_build_response(
                    parsed, context, valid_commands
                )
                
                if mentor_response:
                    mentor_response.tokens_used = tokens_used
                    return mentor_response
                
            except Exception as e:
                if attempt == self.max_retries:
                    # Return a safe default (with token count)
                    fallback = self._get_fallback_command(context, valid_commands)
                    fallback.tokens_used = tokens_used
                    return fallback
        
        # If all retries exhausted, return fallback with accumulated tokens
        fallback = self._get_fallback_command(context, valid_commands)
        fallback.tokens_used = tokens_used
        return fallback
    
    def get_command(
        self,
        context: AttackContext,
        filtered_commands: Optional[List[CommandTemplate]] = None
    ) -> MentorResponse:
        """
        Get the next command synchronously.
        
        Args:
            context: Current attack context
            filtered_commands: Pre-filtered commands (by agent role). If None, uses global registry.
            
        Returns:
            MentorResponse with command and reasoning
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.get_command_async(context, filtered_commands)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(self.get_command_async(context, filtered_commands))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.get_command_async(context, filtered_commands))
    
    def _get_fallback_command(
        self,
        context: AttackContext,
        valid_commands: List[CommandTemplate]
    ) -> MentorResponse:
        """Get a safe fallback command when LLM fails - WITH ANTI-LOOP LOGIC."""
        import random
        
        # Get commands that were ALREADY TRIED (from context history)
        tried_commands = set()
        if context.command_history:
            # Extract base command names (first word) for matching
            for cmd in context.command_history[-20:]:  # Last 20 commands
                # Get the command prefix (e.g., "nmap", "gobuster", "nikto")
                parts = cmd.strip().split()
                if parts:
                    tried_commands.add(parts[0].lower())
        
        if not valid_commands:
            # Ultimate fallback pool - VARIETY of commands to avoid loops
            fallback_pool = [
                (f"nmap -sT --top-ports 100 {context.target}", "nmap_top_ports", "Fast TCP scan"),
                (f"nmap -sV {context.target}", "nmap_version", "Service version detection"),
                (f"masscan -p1-1000 --rate=500 {context.target}", "masscan_fast", "Fast port scan"),
                (f"nmap -sU --top-ports 20 {context.target}", "nmap_udp", "UDP scan"),
                (f"curl -s -I http://{context.target}", "curl_headers", "HTTP headers"),
                (f"dig {context.target} ANY +short", "dig_any", "DNS lookup"),
                (f"whois {context.target} | head -30", "whois", "Domain info"),
            ]
            
            # Filter out commands whose prefix was already tried
            untried = [(c, t, r) for c, t, r in fallback_pool 
                       if c.split()[0].lower() not in tried_commands]
            
            if untried:
                cmd, template_name, reason = random.choice(untried)
            else:
                # All tried - pick a random one anyway (variety within same tools)
                cmd, template_name, reason = random.choice(fallback_pool)
            
            return MentorResponse(
                command=cmd,
                template_name=template_name,
                params={"target": context.target},
                reasoning=f"🔄 Fallback: {reason}",
                confidence=0.3,
                phase=AttackPhase.RECON
            )
        
        # Filter out commands that were already tried (by prefix matching)
        untried_templates = [cmd for cmd in valid_commands 
                            if cmd.name.lower().replace('_', '-').split('_')[0] not in tried_commands
                            and cmd.name.lower().split('_')[0] not in tried_commands]
        
        if untried_templates:
            # Shuffle and pick from untried commands (variety)
            random.shuffle(untried_templates)
            # Weight by reward but not too heavily
            weights = [1.0 + cmd.typical_reward * 0.3 for cmd in untried_templates[:5]]
            template = random.choices(untried_templates[:5], weights=weights[:5], k=1)[0]
        else:
            # All templates tried - just pick randomly for variety
            random.shuffle(valid_commands)
            template = random.choice(valid_commands[:5]) if len(valid_commands) >= 5 else valid_commands[0]
        
        # Try to fill params
        params = {}
        for p in template.required_params:
            if p == "target":
                params[p] = context.target
            elif p == "url":
                params[p] = f"http://{context.target}"
            elif p == "ports" and "open_ports" in context.discoveries:
                params[p] = ",".join(str(p) for p in context.discoveries["open_ports"][:10])
            else:
                params[p] = "{" + p + "}"  # Placeholder
        
        # Add optional defaults
        params.update(template.optional_params)
        
        try:
            command = render_command(template, params)
        except ValueError:
            command = template.template.format_map(params)
        
        return MentorResponse(
            command=command,
            template_name=template.name,
            params=params,
            reasoning=f"Fallback: {template.description}",
            confidence=0.3,
            phase=template.phase
        )
    
    def record_result(
        self,
        response: MentorResponse,
        success: bool,
        reward: float,
        context: AttackContext
    ) -> None:
        """
        Record the result of a command for learning.
        
        Args:
            response: The MentorResponse that was executed
            success: Whether it succeeded
            reward: Reward received
            context: Attack context when executed
        """
        context_tags = {context.platform, context.difficulty}
        preconditions = set(k for k, v in context.state_flags.items() if v)
        
        if success:
            self.learned_store.record_success(
                template_name=response.template_name,
                params=response.params,
                reward=reward,
                context_tags=context_tags,
                preconditions_met=preconditions,
                phase=response.phase
            )
        else:
            self.learned_store.record_failure(
                template_name=response.template_name,
                params=response.params
            )


@dataclass
class DualMentorResponse:
    """
    Response from dual-mentor system with responses from both providers.
    
    Attributes:
        primary: Response from primary mentor (GPT)
        secondary: Response from secondary mentor (Venice)
        chosen: The chosen response to use
        strategy: Strategy used for selection
        provider_used: Which provider's response was used
    """
    primary: Optional[MentorResponse] = None
    secondary: Optional[MentorResponse] = None
    chosen: Optional[MentorResponse] = None
    strategy: str = "gpt_first"
    provider_used: str = "gpt"
    tokens_total: int = 0
    
    @property
    def is_valid(self) -> bool:
        """Check if we have a valid chosen response."""
        return self.chosen is not None and self.chosen.is_valid


class DualMentor:
    """
    Dual-mentor system that leverages both GPT and Venice AI for command generation.
    
    Strategies:
    - gpt_first: Use GPT, fallback to Venice on error
    - venice_first: Use Venice, fallback to GPT on error
    - round_robin: Alternate between GPT and Venice
    - parallel: Query both, choose best by confidence
    - consensus: Query both, require agreement or escalate
    
    Having two mentors enables:
    - Higher mentor call frequency (double the budget)
    - Fallback resilience
    - Different perspectives on attack strategies
    - Uncensored Venice for aggressive tactics
    """
    
    def __init__(
        self,
        gpt_client: Any,
        venice_client: Optional[Any] = None,
        gpt_model: str = "gpt-4o-mini",
        venice_model: str = "venice-uncensored",
        learned_store: Optional[LearnedCommandStore] = None,
        strategy: str = "gpt_first",
        temperature: float = 0.7,
        max_retries: int = 2
    ):
        """
        Initialize the dual-mentor system.
        
        Args:
            gpt_client: OpenAI client for GPT
            venice_client: OpenAI-compatible client for Venice
            gpt_model: GPT model to use
            venice_model: Venice model to use
            learned_store: Store for learned commands
            strategy: Mentor selection strategy
            temperature: Sampling temperature
            max_retries: Max retries for failed parses
        """
        self.learned_store = learned_store or get_learned_store()
        self.strategy = strategy
        self._call_count = 0
        
        # Initialize GPT mentor
        self.gpt_mentor = SmartMentor(
            llm_client=gpt_client,
            learned_store=self.learned_store,
            model=gpt_model,
            temperature=temperature,
            max_retries=max_retries
        ) if gpt_client else None
        
        # Initialize Venice mentor
        self.venice_mentor = SmartMentor(
            llm_client=venice_client,
            learned_store=self.learned_store,
            model=venice_model,
            temperature=temperature + 0.1,  # Venice slightly more creative
            max_retries=max_retries
        ) if venice_client else None
        
        self._has_both = self.gpt_mentor is not None and self.venice_mentor is not None
        
        import logging
        self.logger = logging.getLogger("ariaska.dual_mentor")
        self.logger.info(
            f"DualMentor initialized | Strategy: {strategy} | "
            f"GPT: {'✓' if self.gpt_mentor else '✗'} | "
            f"Venice: {'✓' if self.venice_mentor else '✗'}"
        )
    
    def has_dual_capability(self) -> bool:
        """Check if both mentors are available."""
        return self._has_both
    
    def _get_provider_for_call(self) -> str:
        """Determine which provider to use for this call."""
        if not self._has_both:
            return "gpt" if self.gpt_mentor else "venice"
        
        if self.strategy == "venice_first":
            return "venice"
        elif self.strategy == "round_robin":
            self._call_count += 1
            return "venice" if self._call_count % 2 == 0 else "gpt"
        elif self.strategy in ("parallel", "consensus"):
            return "both"
        else:  # gpt_first (default)
            return "gpt"
    
    async def get_command_async(
        self,
        context: AttackContext,
        filtered_commands: Optional[List[CommandTemplate]] = None,
        force_provider: Optional[str] = None
    ) -> DualMentorResponse:
        """
        Get next command using dual-mentor system asynchronously.
        
        Args:
            context: Current attack context
            filtered_commands: Pre-filtered commands
            force_provider: Force specific provider ("gpt", "venice", "both")
            
        Returns:
            DualMentorResponse with chosen command
        """
        import asyncio
        
        result = DualMentorResponse(strategy=self.strategy)
        provider = force_provider or self._get_provider_for_call()
        
        # === PARALLEL / CONSENSUS: Query both ===
        if provider == "both" and self._has_both:
            try:
                # Run both in parallel
                gpt_task = asyncio.create_task(
                    self.gpt_mentor.get_command_async(context, filtered_commands)  # type: ignore
                )
                venice_task = asyncio.create_task(
                    self.venice_mentor.get_command_async(context, filtered_commands)  # type: ignore
                )
                
                results = await asyncio.gather(
                    gpt_task, venice_task, return_exceptions=True
                )
                
                # Handle exceptions - safely cast to MentorResponse or None
                gpt_resp: Optional[MentorResponse] = None
                venice_resp: Optional[MentorResponse] = None
                
                if isinstance(results[0], MentorResponse):
                    gpt_resp = results[0]
                elif isinstance(results[0], Exception):
                    self.logger.warning(f"GPT mentor error: {results[0]}")
                
                if isinstance(results[1], MentorResponse):
                    venice_resp = results[1]
                elif isinstance(results[1], Exception):
                    self.logger.warning(f"Venice mentor error: {results[1]}")
                
                result.primary = gpt_resp
                result.secondary = venice_resp
                result.tokens_total = (
                    (gpt_resp.tokens_used if gpt_resp else 0) +
                    (venice_resp.tokens_used if venice_resp else 0)
                )
                
                # Choose best response
                if self.strategy == "consensus":
                    result.chosen, result.provider_used = self._consensus_choice(
                        gpt_resp, venice_resp, context
                    )
                else:  # parallel - choose by confidence
                    result.chosen, result.provider_used = self._confidence_choice(
                        gpt_resp, venice_resp
                    )
                
                return result
                
            except Exception as e:
                self.logger.error(f"Parallel query failed: {e}")
                # Fall through to sequential
                provider = "gpt"
        
        # === SEQUENTIAL: Primary then fallback ===
        primary_mentor = self.gpt_mentor if provider == "gpt" else self.venice_mentor
        fallback_mentor = self.venice_mentor if provider == "gpt" else self.gpt_mentor
        
        # Try primary
        if primary_mentor:
            try:
                primary_resp = await primary_mentor.get_command_async(context, filtered_commands)
                result.primary = primary_resp
                result.chosen = primary_resp
                result.provider_used = provider
                result.tokens_total = primary_resp.tokens_used
                
                if primary_resp.is_valid:
                    return result
                    
            except Exception as e:
                self.logger.warning(f"Primary mentor ({provider}) failed: {e}")
        
        # Try fallback
        if fallback_mentor:
            fallback_provider = "venice" if provider == "gpt" else "gpt"
            try:
                fallback_resp = await fallback_mentor.get_command_async(context, filtered_commands)
                result.secondary = fallback_resp
                result.chosen = fallback_resp
                result.provider_used = fallback_provider
                result.tokens_total += fallback_resp.tokens_used
                
                self.logger.info(f"Used fallback mentor: {fallback_provider}")
                return result
                
            except Exception as e:
                self.logger.error(f"Fallback mentor ({fallback_provider}) also failed: {e}")
        
        # Ultimate fallback - return empty response
        return result
    
    def get_command(
        self,
        context: AttackContext,
        filtered_commands: Optional[List[CommandTemplate]] = None,
        force_provider: Optional[str] = None
    ) -> DualMentorResponse:
        """
        Get next command using dual-mentor system synchronously.
        
        Args:
            context: Current attack context
            filtered_commands: Pre-filtered commands
            force_provider: Force specific provider
            
        Returns:
            DualMentorResponse with chosen command
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.get_command_async(context, filtered_commands, force_provider)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self.get_command_async(context, filtered_commands, force_provider)
                )
        except RuntimeError:
            return asyncio.run(
                self.get_command_async(context, filtered_commands, force_provider)
            )
    
    def _confidence_choice(
        self,
        gpt_resp: Optional[MentorResponse],
        venice_resp: Optional[MentorResponse]
    ) -> Tuple[Optional[MentorResponse], str]:
        """Choose response with higher confidence."""
        if not gpt_resp and not venice_resp:
            return None, "none"
        if not gpt_resp:
            return venice_resp, "venice"
        if not venice_resp:
            return gpt_resp, "gpt"
        
        # Both available - choose by confidence
        if venice_resp.confidence > gpt_resp.confidence + 0.1:  # Venice needs 0.1 edge
            return venice_resp, "venice"
        else:
            return gpt_resp, "gpt"
    
    def _consensus_choice(
        self,
        gpt_resp: Optional[MentorResponse],
        venice_resp: Optional[MentorResponse],
        context: AttackContext
    ) -> Tuple[Optional[MentorResponse], str]:
        """Choose based on consensus between mentors."""
        if not gpt_resp and not venice_resp:
            return None, "none"
        if not gpt_resp:
            return venice_resp, "venice"
        if not venice_resp:
            return gpt_resp, "gpt"
        
        # Check if both chose same template
        if gpt_resp.template_name == venice_resp.template_name:
            # Consensus! Use higher confidence
            if venice_resp.confidence > gpt_resp.confidence:
                return venice_resp, "venice_consensus"
            return gpt_resp, "gpt_consensus"
        
        # No consensus - check if commands target same phase
        if gpt_resp.phase == venice_resp.phase:
            # Same phase, use GPT (more conservative)
            return gpt_resp, "gpt_phase_match"
        
        # Different phases - use the one matching context
        if gpt_resp.phase == context.current_phase:
            return gpt_resp, "gpt_phase_aligned"
        if venice_resp.phase == context.current_phase:
            return venice_resp, "venice_phase_aligned"
        
        # Default to GPT (primary)
        return gpt_resp, "gpt_default"
    
    def record_result(
        self,
        response: MentorResponse,
        success: bool,
        reward: float,
        context: AttackContext
    ) -> None:
        """Record result to learned store (shared between mentors)."""
        if self.gpt_mentor:
            self.gpt_mentor.record_result(response, success, reward, context)


def create_smart_mentor(
    llm_client: Any,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7
) -> SmartMentor:
    """
    Factory function to create a SmartMentor.
    
    Args:
        llm_client: LLM client with chat completion capability
        model: Model to use
        temperature: Sampling temperature
        
    Returns:
        Configured SmartMentor instance
    """
    return SmartMentor(
        llm_client=llm_client,
        model=model,
        temperature=temperature
    )


def create_dual_mentor(
    gpt_client: Any,
    venice_client: Optional[Any] = None,
    gpt_model: str = "gpt-4o-mini",
    venice_model: str = "venice-uncensored",
    strategy: str = "gpt_first",
    temperature: float = 0.7
) -> DualMentor:
    """
    Factory function to create a DualMentor.
    
    Args:
        gpt_client: OpenAI client for GPT
        venice_client: OpenAI-compatible client for Venice (optional)
        gpt_model: GPT model to use
        venice_model: Venice model to use
        strategy: Mentor selection strategy (gpt_first, venice_first, round_robin, parallel, consensus)
        temperature: Sampling temperature
        
    Returns:
        Configured DualMentor instance
    """
    return DualMentor(
        gpt_client=gpt_client,
        venice_client=venice_client,
        gpt_model=gpt_model,
        venice_model=venice_model,
        strategy=strategy,
        temperature=temperature
    )
