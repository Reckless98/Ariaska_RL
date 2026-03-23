"""Strategic LLM Advisor — Multi-step planning for PPO/DDQN guidance.

Instead of calling the LLM every step for tactical command selection,
the Strategic Advisor is called every N steps to produce a multi-step
action plan. PPO/DDQN execute the plan autonomously between LLM calls.

Architecture:
    LLM called every plan_horizon steps → produces ordered action list
    SmartCoach checks plan before calling micro_chain
    If plan step available → use directly (zero LLM latency)
    If plan exhausted/stale → trigger re-planning

This reduces LLM calls by ~80% while using the LLM for what it's good
at: strategic reasoning over multiple steps.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.gpt_manager import GPTManager

logger = logging.getLogger("ariaska.llm.strategic_advisor")


@dataclass
class PlanStep:
    """One step in a strategic plan."""
    template_name: str
    command: str
    reasoning: str
    priority: float = 0.5
    executed: bool = False
    skipped: bool = False
    step_created: int = 0


@dataclass
class StrategicPlan:
    """Multi-step action plan produced by the LLM."""
    steps: List[PlanStep] = field(default_factory=list)
    created_at_step: int = 0
    valid_until_step: int = 0
    phase_at_creation: str = ""
    creation_time: float = 0.0
    llm_latency_ms: int = 0
    plan_id: int = 0

    @property
    def remaining_steps(self) -> List[PlanStep]:
        return [s for s in self.steps if not s.executed and not s.skipped]

    @property
    def is_exhausted(self) -> bool:
        return len(self.remaining_steps) == 0

    @property
    def execution_rate(self) -> float:
        if not self.steps:
            return 0.0
        executed = sum(1 for s in self.steps if s.executed)
        return executed / len(self.steps)


class StrategicAdvisor:
    """LLM-powered strategic planner for multi-step action sequences.

    Called every `plan_horizon` steps to produce an ordered list of
    recommended actions. PPO/DDQN use these as guidance between LLM calls.

    Usage::

        advisor = StrategicAdvisor(gpt_manager, plan_horizon=5)

        # At each step in SmartCoach:
        if advisor.should_replan(step, phase, discovery_board):
            advisor.create_plan(phase, discovery_board, templates, agent_role)

        next_step = advisor.get_next_step(available_templates)
        if next_step:
            # Use plan step directly — zero LLM latency
            command = next_step.command
        else:
            # Fall through to micro_chain
            ...
    """

    # Phases where replanning triggers more aggressively
    _VOLATILE_PHASES = frozenset({
        "EXPLOITATION", "PRIVILEGE_ESCALATION", "POST_EXPLOITATION",
    })

    def __init__(
        self,
        gpt_manager: "GPTManager",
        plan_horizon: int = 5,
        replan_on_phase_change: bool = True,
        replan_on_stagnation: int = 3,
    ) -> None:
        self._gpt = gpt_manager
        self.plan_horizon = plan_horizon
        self.replan_on_phase_change = replan_on_phase_change
        self.replan_on_stagnation = replan_on_stagnation

        self._current_plan: Optional[StrategicPlan] = None
        self._plan_count: int = 0
        self._last_plan_step: int = -999
        self._last_plan_phase: str = ""
        self._total_plans: int = 0
        self._total_plan_steps_executed: int = 0
        self._total_plan_steps_skipped: int = 0

        logger.info(
            "StrategicAdvisor initialized | horizon=%d | phase_replan=%s | stag_replan=%d",
            plan_horizon, replan_on_phase_change, replan_on_stagnation,
        )

    def should_replan(
        self,
        step: int,
        phase: str,
        stagnation: int = 0,
    ) -> bool:
        """Check if a new strategic plan is needed."""
        # No plan yet
        if self._current_plan is None:
            return True

        # Plan exhausted
        if self._current_plan.is_exhausted:
            return True

        # Plan expired (past valid_until)
        if step > self._current_plan.valid_until_step:
            return True

        # Phase changed since last plan
        if self.replan_on_phase_change and phase != self._last_plan_phase:
            logger.info(
                f"[STRATEGIC] Phase changed {self._last_plan_phase} → {phase}, replanning"
            )
            return True

        # Stagnation threshold hit
        if stagnation >= self.replan_on_stagnation and step - self._last_plan_step >= 2:
            logger.info(
                f"[STRATEGIC] Stagnation={stagnation} >= {self.replan_on_stagnation}, replanning"
            )
            return True

        return False

    def create_plan(
        self,
        phase: str,
        discovery_board: Dict[str, Any],
        available_templates: List[str],
        agent_role: str,
        step: int = 0,
        recent_commands: Optional[List[str]] = None,
    ) -> Optional[StrategicPlan]:
        """Create a multi-step strategic plan via single LLM call."""
        if not self._gpt.can_make_request():
            logger.debug("[STRATEGIC] Budget exhausted, no plan created")
            return None

        recent = recent_commands[-8:] if recent_commands else []
        ports = sorted(str(p) for p in list(discovery_board.get("ports", []))[:12])
        services = sorted(str(s) for s in list(discovery_board.get("services", []))[:10])
        creds = list(discovery_board.get("credentials", []))[:5]
        shells = list(discovery_board.get("shells", []))[:3]
        vulns = list(discovery_board.get("vulns", []))[:5]
        web_paths = list(discovery_board.get("web_paths", []))[:8]
        templates = available_templates[:15]

        prompt = (
            f"You are a strategic cybersecurity advisor planning an engagement.\n"
            f"Phase: {phase} | Role: {agent_role}\n"
            f"Ports: {ports}\n"
            f"Services: {services}\n"
            f"Credentials: {creds}\n"
            f"Shells: {len(shells)} active\n"
            f"Vulns: {vulns}\n"
            f"Web paths: {web_paths}\n"
            f"Recent commands (avoid repeating): {recent}\n"
            f"Available templates: {templates}\n\n"
            f"Plan the next {self.plan_horizon} actions in priority order. "
            f"Each action should build on the previous — create a coherent attack sequence.\n"
            f"Focus on the biggest gaps: what haven't we tried yet? What services are unexplored?\n\n"
            f'JSON array: [{{"template":"name","command":"full command","reasoning":"why this now","priority":0.0-1.0}},...]\n'
            f"Return ONLY the JSON array, {self.plan_horizon} items."
        )

        t0 = time.monotonic()
        try:
            response = self._gpt.gpt_request(
                prompt=prompt,
                task_type="strategic",
                agent_id="strategic_advisor",
                max_tokens=2400,  # P58: 1800→2400 — room for think blocks + full 20-step plan
                model="local-llm",
                timeout=30,
            )
            latency_ms = int((time.monotonic() - t0) * 1000)

            plan_steps = self._parse_plan(response, step, templates)
            if not plan_steps:
                logger.warning(f"[STRATEGIC] Failed to parse plan from LLM response")
                return None

            self._plan_count += 1
            plan = StrategicPlan(
                steps=plan_steps,
                created_at_step=step,
                valid_until_step=step + self.plan_horizon + 2,  # buffer
                phase_at_creation=phase,
                creation_time=time.time(),
                llm_latency_ms=latency_ms,
                plan_id=self._plan_count,
            )

            self._current_plan = plan
            self._last_plan_step = step
            self._last_plan_phase = phase
            self._total_plans += 1

            logger.info(
                f"[STRATEGIC] Plan #{self._plan_count} created: "
                f"{len(plan_steps)} steps, latency={latency_ms}ms, "
                f"phase={phase}, templates={[s.template_name for s in plan_steps]}"
            )
            return plan

        except Exception as e:
            logger.warning(f"[STRATEGIC] Plan creation failed: {e}")
            return None

    def get_next_step(
        self,
        available_templates: Optional[List[str]] = None,
    ) -> Optional[PlanStep]:
        """Get the next unexecuted step from the current plan.

        If `available_templates` is provided, skip plan steps whose
        template_name isn't in the available set.
        """
        if self._current_plan is None:
            return None

        available_set = set(available_templates) if available_templates else None

        for plan_step in self._current_plan.steps:
            if plan_step.executed or plan_step.skipped:
                continue

            # Skip if template not available in current context
            if available_set and plan_step.template_name not in available_set:
                plan_step.skipped = True
                self._total_plan_steps_skipped += 1
                logger.debug(
                    f"[STRATEGIC] Skipping plan step '{plan_step.template_name}' "
                    f"— not in available templates"
                )
                continue

            return plan_step

        return None

    def mark_executed(self, plan_step: PlanStep) -> None:
        """Mark a plan step as executed."""
        plan_step.executed = True
        self._total_plan_steps_executed += 1

    def invalidate_plan(self, reason: str) -> None:
        """Force plan invalidation (e.g., on major discovery)."""
        if self._current_plan is not None:
            logger.info(
                f"[STRATEGIC] Plan #{self._current_plan.plan_id} invalidated: {reason}"
            )
            self._current_plan = None

    def _parse_plan(
        self,
        response: str,
        step: int,
        available_templates: List[str],
    ) -> List[PlanStep]:
        """Parse LLM response into PlanStep list."""
        if not response:
            return []

        # Strip markdown/think wrappers
        text = response.strip()
        for prefix in ("```json", "```"):
            if text.startswith(prefix):
                text = text[len(prefix):]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Find JSON array
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            # Try extracting individual JSON objects
            return self._parse_individual_objects(text, step)

        try:
            items = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return self._parse_individual_objects(text, step)

        if not isinstance(items, list):
            return []

        plan_steps = []
        for item in items[:self.plan_horizon]:
            if not isinstance(item, dict):
                continue
            template = str(item.get("template", item.get("template_name", "")))
            command = str(item.get("command", ""))
            reasoning = str(item.get("reasoning", ""))
            priority = max(0.0, min(1.0, float(item.get("priority", 0.5))))

            if not command and not template:
                continue

            plan_steps.append(PlanStep(
                template_name=template,
                command=command,
                reasoning=reasoning,
                priority=priority,
                step_created=step,
            ))

        return plan_steps

    def _parse_individual_objects(self, text: str, step: int) -> List[PlanStep]:
        """Fallback: extract individual JSON objects from text."""
        import re
        objects = re.findall(r'\{[^{}]+\}', text)
        plan_steps = []
        for obj_str in objects[:self.plan_horizon]:
            try:
                item = json.loads(obj_str)
                template = str(item.get("template", item.get("template_name", "")))
                command = str(item.get("command", ""))
                if command or template:
                    plan_steps.append(PlanStep(
                        template_name=template,
                        command=command,
                        reasoning=str(item.get("reasoning", "")),
                        priority=max(0.0, min(1.0, float(item.get("priority", 0.5)))),
                        step_created=step,
                    ))
            except (json.JSONDecodeError, ValueError):
                continue
        return plan_steps

    def get_stats(self) -> Dict[str, Any]:
        """Return advisor statistics."""
        plan_info = None
        if self._current_plan is not None:
            plan_info = {
                "plan_id": self._current_plan.plan_id,
                "total_steps": len(self._current_plan.steps),
                "remaining": len(self._current_plan.remaining_steps),
                "execution_rate": round(self._current_plan.execution_rate, 2),
                "phase": self._current_plan.phase_at_creation,
                "latency_ms": self._current_plan.llm_latency_ms,
            }
        return {
            "total_plans": self._total_plans,
            "steps_executed": self._total_plan_steps_executed,
            "steps_skipped": self._total_plan_steps_skipped,
            "current_plan": plan_info,
            "plan_horizon": self.plan_horizon,
        }
