#!/usr/bin/env python3
"""
core/training/apprentice_coach.py — ARIASKA Apprentice Coach System v1.0

Shared mentor wrapper for all agents. Handles GPT consultation, skill extraction,
and decision recording for the apprentice-to-autonomy training paradigm.
"""

import time
import logging
import hashlib
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from dataclasses import dataclass, field

from core.training.mentor_policy import MentorPolicy, MentorPolicyConfig

if TYPE_CHECKING:
    from core.gpt_manager import GPTManager
    from core.postmortem import SkillLibrary
    from core.tracing import TraceWriter

logger = logging.getLogger("ariaska.apprentice_coach")


@dataclass
class DecisionResult:
    """Result of a coach decision."""
    chosen_action: str
    mentor_call: bool
    model_used: Optional[str] = None
    mentor_guidance: Optional[str] = None
    mentor_delta: str = "kept"  # "kept" or "changed"
    skill_cards: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)


@dataclass
class StepContext:
    """Context for a single step decision."""
    episode: int
    step: int
    phase: str
    state: Dict[str, Any]
    agent_name: str
    
    # Optional enrichment
    history: List[str] = field(default_factory=list)
    target_ip: str = "10.10.10.10"
    open_ports: List[int] = field(default_factory=list)
    detection_risk: float = 0.0
    blue_alert_level: float = 0.0


class ApprenticeCoach:
    """
    Shared mentor wrapper for all agents.
    
    Responsibilities:
    - Decide when to call mentor (via MentorPolicy)
    - Generate prompts for each agent type
    - Parse mentor responses into guidance and skill cards
    - Track decisions for logging
    
    Works in both online (GPT) and offline (deterministic placeholder) modes.
    """
    
    # Agent-specific prompt templates
    AGENT_PROMPTS = {
        "ScoutAgent": """You are advising ScoutAgent, a reconnaissance specialist.
Current phase: {phase}
Target: {target_ip}
Known ports: {open_ports}
Detection risk: {detection_risk}

Scout proposes: {proposed_action}
Scout confidence: {confidence:.2f}

If this is suboptimal, suggest a better recon approach. Be concise.
Format: ACTION: <command> | REASON: <brief reason>""",

        "RedAgent": """You are advising RedAgent, an offensive security specialist.
Current phase: {phase}
Target: {target_ip}
Open ports: {open_ports}
Detection risk: {detection_risk}

Red proposes: {proposed_action}
Red confidence: {confidence:.2f}

If this is suboptimal, suggest a better attack approach. Be concise.
Format: ACTION: <command> | REASON: <brief reason>""",

        "BlueAgent": """You are advising BlueAgent, a defensive security specialist.
Current phase: {phase}
Alert level: {blue_alert_level}
Detection risk: {detection_risk}

Blue proposes: {proposed_action}
Blue confidence: {confidence:.2f}

If this is suboptimal, suggest a better defensive action. Be concise.
Format: ACTION: <action> | REASON: <brief reason>""",

        "OrionAgent": """You are advising OrionAgent, the strategic coordinator.
Current phase: {phase}
Recent actions: {history}
Detection risk: {detection_risk}

Orion proposes critique/suggestion: {proposed_action}
Orion confidence: {confidence:.2f}

Provide strategic guidance. Be concise.
Format: GUIDANCE: <strategic advice> | PRIORITY: <high/medium/low>""",

        "ShadowAgent": """You are advising ShadowAgent, the stealth monitor.
Current phase: {phase}
Detection risk: {detection_risk}
Alert level: {blue_alert_level}

Shadow proposes: {proposed_action}
Shadow confidence: {confidence:.2f}

Advise on what to remember or flag for memory. Be concise.
Format: MEMORY: <what to store> | FLAG: <risk level>""",
    }
    
    # Offline placeholders per agent
    OFFLINE_PLACEHOLDERS = {
        "ScoutAgent": "ACTION: nmap -sV -T2 {target_ip} | REASON: Offline mode placeholder",
        "RedAgent": "ACTION: gobuster dir -u http://{target_ip} | REASON: Offline mode placeholder",
        "BlueAgent": "ACTION: monitor_network | REASON: Offline mode placeholder",
        "OrionAgent": "GUIDANCE: Continue current strategy | PRIORITY: medium",
        "ShadowAgent": "MEMORY: Step recorded | FLAG: low",
    }
    
    def __init__(
        self,
        agent_name: str,
        gpt_manager: "GPTManager",
        mentor_policy: Optional[MentorPolicy] = None,
        skill_library: Optional["SkillLibrary"] = None,
        trace_writer: Optional["TraceWriter"] = None,
        mentor_log_path: Optional[str] = None,
    ):
        self.agent_name = agent_name
        self.gpt_manager = gpt_manager
        self.mentor_policy = mentor_policy or MentorPolicy()
        self.skill_library = skill_library
        self.trace_writer = trace_writer
        self.mentor_log_path = mentor_log_path
        
        # Decision history for this coach
        self.decisions: List[DecisionResult] = []
        
        logger.debug(f"ApprenticeCoach initialized for {agent_name}")
    
    def decide(
        self,
        step_ctx: StepContext,
        proposed_action: str,
        confidence: Optional[float] = None,
    ) -> DecisionResult:
        """
        Make a decision: keep agent's proposal or consult mentor.
        
        Args:
            step_ctx: Context for this step
            proposed_action: Agent's proposed action
            confidence: Agent's confidence (0-1), uses default if None
            
        Returns:
            DecisionResult with chosen action and metadata
        """
        confidence = confidence if confidence is not None else 0.5
        
        # Check skill library for matching rules
        skill_match = self._check_skill_library(step_ctx, proposed_action)
        if skill_match:
            logger.debug(f"{self.agent_name}: Skill match found, using library action")
            return DecisionResult(
                chosen_action=skill_match["action"],
                mentor_call=False,
                mentor_delta="changed" if skill_match["action"] != proposed_action else "kept",
                skill_cards=[skill_match],
                confidence=confidence,
            )
        
        # Check if mentor should be called
        should_call = self.mentor_policy.should_call_mentor(
            agent_name=self.agent_name,
            confidence=confidence,
            episode=step_ctx.episode,
            step=step_ctx.step,
        )
        
        if not should_call:
            # Keep agent's action
            result = DecisionResult(
                chosen_action=proposed_action,
                mentor_call=False,
                mentor_delta="kept",
                confidence=confidence,
            )
            self.decisions.append(result)
            return result
        
        # Call mentor
        result = self._call_mentor(step_ctx, proposed_action, confidence)
        self.decisions.append(result)
        
        # Log to mentor transcript
        if self.mentor_log_path:
            self._log_mentor_call(step_ctx, proposed_action, result)
        
        return result
    
    def _call_mentor(
        self,
        step_ctx: StepContext,
        proposed_action: str,
        confidence: float,
    ) -> DecisionResult:
        """Call GPT mentor for guidance."""
        
        # Check if offline
        if self.gpt_manager.is_offline():
            return self._get_offline_result(step_ctx, proposed_action, confidence)
        
        # Build prompt
        prompt_template = self.AGENT_PROMPTS.get(
            self.agent_name,
            self.AGENT_PROMPTS["RedAgent"]  # Fallback
        )
        
        prompt = prompt_template.format(
            phase=step_ctx.phase,
            target_ip=step_ctx.target_ip,
            open_ports=step_ctx.open_ports,
            detection_risk=step_ctx.detection_risk,
            blue_alert_level=step_ctx.blue_alert_level,
            proposed_action=proposed_action,
            confidence=confidence,
            history="; ".join(step_ctx.history[-3:]) if step_ctx.history else "none",
        )
        
        # Use unified request method for proper token tracking
        start_time = time.time()
        result = self.gpt_manager.request(
            role=self.agent_name,
            task_type="tactical",
            prompt=prompt,
            max_tokens=150,
        )
        latency_ms = int((time.time() - start_time) * 1000)
        
        if result["success"]:
            response = result["response"]
            model_used = result["model_used"]
            
            # Parse response
            chosen_action, guidance = self._parse_mentor_response(
                response, proposed_action, step_ctx
            )
            
            # Extract skill cards
            skill_cards = self._extract_skill_cards(
                step_ctx, proposed_action, chosen_action, guidance
            )
            
            return DecisionResult(
                chosen_action=chosen_action,
                mentor_call=True,
                model_used=model_used,
                mentor_guidance=guidance,
                mentor_delta="changed" if chosen_action != proposed_action else "kept",
                skill_cards=skill_cards,
                confidence=confidence,
            )
        else:
            # Request failed (budget exceeded or API error)
            error = result.get("error", "Unknown error")
            logger.warning(f"Mentor call failed for {self.agent_name}: {error}")
            return DecisionResult(
                chosen_action=proposed_action,
                mentor_call=True,
                model_used=result.get("model_used"),
                mentor_guidance=f"Error: {str(error)[:50]}",
                mentor_delta="kept",
                confidence=confidence,
            )
    
    def _get_offline_result(
        self,
        step_ctx: StepContext,
        proposed_action: str,
        confidence: float,
    ) -> DecisionResult:
        """Generate varied offline result using anti-loop logic."""
        import random
        
        # Get command history from step_ctx
        history = step_ctx.history if step_ctx.history else []
        # Also check state for command_history
        state = step_ctx.state if hasattr(step_ctx, 'state') else {}
        full_history = history + state.get("command_history", [])
        
        # Extract tried prefixes
        tried_prefixes = set()
        for cmd in full_history[-15:]:
            if isinstance(cmd, str):
                parts = cmd.strip().split()
                if parts:
                    tried_prefixes.add(parts[0].lower())
        
        target = step_ctx.target_ip
        
        # Agent-specific command pools - VARIED offline fallbacks
        agent_commands = {
            "ScoutAgent": [
                (f"nmap -sV -T2 {target}", "nmap"),
                (f"nmap -sT --top-ports 100 {target}", "nmap"),
                (f"nmap -A {target}", "nmap"),
                (f"masscan -p1-1000 --rate=500 {target}", "masscan"),
                (f"rustscan -a {target} --ulimit 5000", "rustscan"),
                (f"dig {target} ANY +noall +answer", "dig"),
                (f"whois {target}", "whois"),
                (f"whatweb http://{target}", "whatweb"),
                (f"curl -sI http://{target}", "curl"),
                (f"wafw00f http://{target}", "wafw00f"),
            ],
            "RedAgent": [
                (f"gobuster dir -u http://{target} -w /usr/share/wordlists/dirb/common.txt", "gobuster"),
                (f"nikto -h http://{target}", "nikto"),
                (f"sqlmap -u 'http://{target}/?id=1' --batch", "sqlmap"),
                (f"nuclei -u http://{target} -t cves/", "nuclei"),
                (f"hydra -l admin -P /usr/share/wordlists/rockyou.txt ssh://{target} -t 4", "hydra"),
                (f"crackmapexec smb {target} --shares", "crackmapexec"),
                (f"searchsploit linux kernel 5", "searchsploit"),
                (f"wpscan --url http://{target} --enumerate u", "wpscan"),
                (f"ffuf -w /usr/share/wordlists/dirb/common.txt -u http://{target}/FUZZ", "ffuf"),
                (f"feroxbuster -u http://{target}", "feroxbuster"),
            ],
            "BlueAgent": [
                ("monitor_network", "monitor"),
                ("check_logs", "check"),
                ("analyze_traffic", "analyze"),
                ("scan_malware", "scan"),
                ("review_connections", "review"),
            ],
            "OrionAgent": [
                ("GUIDANCE: Continue current strategy | PRIORITY: medium", "guidance"),
                ("GUIDANCE: Shift to enumeration phase | PRIORITY: high", "guidance"),
                ("GUIDANCE: Focus on discovered services | PRIORITY: medium", "guidance"),
            ],
            "ShadowAgent": [
                ("MEMORY: Step recorded | FLAG: low", "memory"),
                ("MEMORY: Notable finding logged | FLAG: medium", "memory"),
                ("MEMORY: Pattern detected | FLAG: low", "memory"),
            ],
        }
        
        commands = agent_commands.get(self.agent_name, [])
        
        if commands:
            # Filter out already-tried prefixes for action agents
            if self.agent_name in ["ScoutAgent", "RedAgent", "BlueAgent"]:
                untried = [(cmd, prefix) for cmd, prefix in commands if prefix.lower() not in tried_prefixes]
                if untried:
                    action_part, _ = random.choice(untried)
                else:
                    # All tried - pick random from full list
                    action_part, _ = random.choice(commands)
            else:
                # OrionAgent, ShadowAgent - just rotate
                action_part, _ = random.choice(commands)
            
            placeholder = f"ACTION: {action_part} | REASON: Offline mode varied"
        else:
            placeholder = self.OFFLINE_PLACEHOLDERS.get(
                self.agent_name,
                "ACTION: {proposed} | REASON: Offline mode"
            ).format(
                target_ip=step_ctx.target_ip,
                proposed=proposed_action,
            )
            action_part = proposed_action
        
        # Parse action for actual commands
        if self.agent_name in ["OrionAgent", "ShadowAgent"]:
            # These don't use ACTION: format in the same way
            if "GUIDANCE:" in action_part:
                action_part = "analyze_environment"
                placeholder = action_part  # Keep original format
            elif "MEMORY:" in action_part:
                action_part = "observe_quietly"
                placeholder = action_part
        
        return DecisionResult(
            chosen_action=action_part,
            mentor_call=True,
            model_used="offline",
            mentor_guidance=placeholder,
            mentor_delta="changed" if action_part != proposed_action else "kept",
            confidence=confidence,
        )
    
    def _parse_mentor_response(
        self,
        response: str,
        proposed_action: str,
        step_ctx: StepContext,
    ) -> tuple:
        """Parse mentor response into action and guidance."""
        if not response:
            return proposed_action, "No response"
        
        guidance = response.strip()
        chosen_action = proposed_action
        
        # Try to extract ACTION
        if "ACTION:" in response:
            try:
                action_part = response.split("ACTION:")[1]
                if "|" in action_part:
                    action_part = action_part.split("|")[0]
                chosen_action = action_part.strip()
            except:
                pass
        elif "GUIDANCE:" in response:
            # OrionAgent uses GUIDANCE format - keep proposed action
            pass
        elif "MEMORY:" in response:
            # ShadowAgent uses MEMORY format - keep proposed action
            pass
        
        return chosen_action, guidance
    
    def _extract_skill_cards(
        self,
        step_ctx: StepContext,
        proposed_action: str,
        chosen_action: str,
        guidance: str,
    ) -> List[Dict[str, Any]]:
        """Extract skill cards from mentor interaction."""
        if chosen_action == proposed_action:
            return []  # No new skill if action wasn't changed
        
        # Generate skill ID
        skill_id = hashlib.md5(
            f"{self.agent_name}:{step_ctx.phase}:{chosen_action}".encode()
        ).hexdigest()[:12]
        
        skill_card = {
            "id": f"skill_{skill_id}",
            "agent": self.agent_name,
            "if_condition": f"phase={step_ctx.phase}",
            "then_action": chosen_action,
            "confidence": 0.7,  # Initial confidence for new skill
            "tags": [step_ctx.phase, self.agent_name.lower()],
            "source": "mentor",
        }
        
        # Store in skill library if available
        if self.skill_library:
            try:
                from core.postmortem import SkillCard
                card = SkillCard(
                    id=skill_card["id"],
                    if_condition=skill_card["if_condition"],
                    then_action=skill_card["then_action"],
                    confidence=skill_card["confidence"],
                    tags=skill_card["tags"],
                )
                self.skill_library.promote(card, reason=f"Mentor guidance for {self.agent_name}")
            except Exception as e:
                logger.debug(f"Failed to store skill: {e}")
        
        return [skill_card]
    
    def _check_skill_library(
        self,
        step_ctx: StepContext,
        proposed_action: str,
    ) -> Optional[Dict[str, Any]]:
        """Check skill library for matching rules."""
        if not self.skill_library:
            return None
        
        try:
            # Simple phase-based matching
            all_skills = self.skill_library.get_all_skills()
            for skill in all_skills:
                # Check if skill matches current phase and agent
                if hasattr(skill, 'if_condition') and hasattr(skill, 'then_action'):
                    condition = skill.if_condition.lower()
                    if f"phase={step_ctx.phase}" in condition.lower():
                        # Check agent tag if present
                        tags = getattr(skill, 'tags', []) or []
                        if not tags or self.agent_name.lower() in [t.lower() for t in tags]:
                            return {
                                "id": skill.id,
                                "action": skill.then_action,
                                "confidence": getattr(skill, 'confidence', 0.5),
                            }
        except Exception as e:
            logger.debug(f"Skill library check failed: {e}")
        
        return None
    
    def _log_mentor_call(
        self,
        step_ctx: StepContext,
        proposed_action: str,
        result: DecisionResult,
    ):
        """Log mentor call to transcript file."""
        import json
        
        entry = {
            "event_id": f"{step_ctx.episode}:{step_ctx.step:04d}:{self.agent_name}",
            "agent": self.agent_name,
            "task_type": "tactical",
            "prompt_summary": f"{step_ctx.phase}: {proposed_action[:50]}",
            "mentor_guidance": result.mentor_guidance[:200] if result.mentor_guidance else None,
            "model_used": result.model_used,
            "mentor_delta": result.mentor_delta,
            "timestamp": result.timestamp,
        }
        
        try:
            import os
            os.makedirs(os.path.dirname(self.mentor_log_path), exist_ok=True)
            with open(self.mentor_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.debug(f"Failed to log mentor call: {e}")
    
    def reset_episode(self, episode: int):
        """Reset for new episode."""
        self.mentor_policy.reset_episode(episode)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get coach statistics."""
        return {
            "agent": self.agent_name,
            "total_decisions": len(self.decisions),
            "mentor_calls": sum(1 for d in self.decisions if d.mentor_call),
            "actions_changed": sum(1 for d in self.decisions if d.mentor_delta == "changed"),
            "policy_stats": self.mentor_policy.get_stats(),
        }
