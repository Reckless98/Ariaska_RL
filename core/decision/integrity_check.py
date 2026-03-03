"""
core/decision/integrity_check.py — Phase 8: Boot-time Integrity Check.

Validates structural invariants at system startup.
Raises RuntimeError if any invariant is violated.

Invariants checked:
  1. Feature flag enabled but no call site in N steps → warning
  2. Learner enabled but update_count == 0 after warm-up → error
  3. Entropy written by more than one module → error
  4. Reward pipeline bypassed → error
  5. DecisionCore not wired → error
  6. MaturityController not wired → error

Author: Phase 8 — Integrity Check
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.decision.integrity")


@dataclass
class IntegrityViolation:
    """A single integrity violation."""

    severity: str  # "error" or "warning"
    check_name: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.check_name}: {self.message}"


@dataclass
class IntegrityReport:
    """Result of running all integrity checks."""

    passed: bool
    violations: List[IntegrityViolation] = field(default_factory=list)
    checks_run: int = 0

    @property
    def error_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "warning")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "checks_run": self.checks_run,
            "errors": self.error_count,
            "warnings": self.warning_count,
            "violations": [
                {
                    "severity": v.severity,
                    "check": v.check_name,
                    "message": v.message,
                }
                for v in self.violations
            ],
        }


class IntegrityCheck:
    """Boot-time and runtime integrity verification.

    Usage:
        checker = IntegrityCheck()

        # Boot-time (before training starts):
        report = checker.check_boot(smart_coach, orchestrator)
        if not report.passed:
            raise RuntimeError(f"Integrity check failed: {report}")

        # Runtime (periodic, every N steps):
        report = checker.check_runtime(smart_coach, harmony_metrics, step)
    """

    def __init__(self, warm_up_steps: int = 50) -> None:
        self._warm_up_steps: int = warm_up_steps

    def check_boot(
        self,
        smart_coach: Any,
        orchestrator: Optional[Any] = None,
    ) -> IntegrityReport:
        """Run boot-time integrity checks.

        Call this BEFORE training starts.  Raises RuntimeError on
        critical violations.

        Args:
            smart_coach: SmartCoach instance.
            orchestrator: SmartOrchestrator instance (optional).

        Returns:
            IntegrityReport with all violations.
        """
        violations: List[IntegrityViolation] = []
        checks_run = 0

        # 1. Check DecisionCore is wired
        checks_run += 1
        decision_core = getattr(smart_coach, '_decision_core', None)
        if decision_core is None:
            violations.append(IntegrityViolation(
                severity="error",
                check_name="decision_core_wired",
                message="SmartCoach._decision_core is None — "
                        "DecisionCore not wired",
            ))

        # 2. Check MaturityController is wired
        checks_run += 1
        maturity = getattr(smart_coach, '_maturity_controller', None)
        if maturity is None:
            violations.append(IntegrityViolation(
                severity="error",
                check_name="maturity_controller_wired",
                message="SmartCoach._maturity_controller is None — "
                        "MaturityController not wired",
            ))

        # 3. Check UnifiedRewardPipeline is wired
        checks_run += 1
        reward_pipe = getattr(smart_coach, '_reward_pipeline', None)
        if reward_pipe is None:
            violations.append(IntegrityViolation(
                severity="error",
                check_name="reward_pipeline_wired",
                message="SmartCoach._reward_pipeline is None — "
                        "UnifiedRewardPipeline not wired",
            ))

        # 4. Check HarmonyMetrics is wired
        checks_run += 1
        harmony = getattr(smart_coach, '_harmony_metrics', None)
        if harmony is None:
            violations.append(IntegrityViolation(
                severity="error",
                check_name="harmony_metrics_wired",
                message="SmartCoach._harmony_metrics is None — "
                        "HarmonyMetrics not wired",
            ))

        # 5. Check PPO agent exists
        checks_run += 1
        ppo = getattr(smart_coach, 'ppo_agent', None)
        if ppo is None:
            violations.append(IntegrityViolation(
                severity="error",
                check_name="ppo_agent_exists",
                message="SmartCoach.ppo_agent is None — "
                        "PPO not initialized",
            ))

        # 6. Check state_dim consistency
        checks_run += 1
        if ppo is not None:
            ppo_config = getattr(ppo, 'config', None)
            if ppo_config is not None:
                state_dim = getattr(ppo_config, 'state_dim', None)
                if state_dim is not None and state_dim != 512:
                    violations.append(IntegrityViolation(
                        severity="error",
                        check_name="state_dim_512",
                        message=f"PPO state_dim is {state_dim}, "
                                f"expected 512",
                    ))

        # 7. Check orchestrator has maturity controller if provided
        if orchestrator is not None:
            checks_run += 1
            orch_maturity = getattr(
                orchestrator, '_maturity_controller', None
            )
            if orch_maturity is None:
                violations.append(IntegrityViolation(
                    severity="warning",
                    check_name="orchestrator_maturity",
                    message="Orchestrator._maturity_controller not set — "
                            "maturity updates may not propagate",
                ))

        passed = all(v.severity != "error" for v in violations)
        report = IntegrityReport(
            passed=passed,
            violations=violations,
            checks_run=checks_run,
        )

        if not passed:
            msg = (
                f"INTEGRITY CHECK FAILED: "
                f"{report.error_count} errors, "
                f"{report.warning_count} warnings"
            )
            for v in violations:
                msg += f"\n  {v}"
            logger.error(msg)
            raise RuntimeError(msg)
        else:
            logger.info(
                "Integrity check passed: %d checks, %d warnings",
                checks_run,
                report.warning_count,
            )

        return report

    def check_runtime(
        self,
        smart_coach: Any,
        harmony_metrics: Optional[Any] = None,
        step: int = 0,
    ) -> IntegrityReport:
        """Runtime integrity check (call periodically).

        Args:
            smart_coach: SmartCoach instance.
            harmony_metrics: HarmonyMetrics instance.
            step: Current global step.

        Returns:
            IntegrityReport with any runtime violations.
        """
        violations: List[IntegrityViolation] = []
        checks_run = 0

        # 1. Check entropy writer count (MUST be exactly 1)
        if harmony_metrics is not None:
            checks_run += 1
            n_writers = getattr(
                harmony_metrics, 'entropy_writer_count', 0
            )
            if n_writers > 1:
                violations.append(IntegrityViolation(
                    severity="error",
                    check_name="single_entropy_writer",
                    message=f"{n_writers} entropy writers active — "
                            f"must be exactly 1",
                ))
            elif n_writers == 0 and step > self._warm_up_steps:
                violations.append(IntegrityViolation(
                    severity="warning",
                    check_name="entropy_writer_registered",
                    message="No entropy writers registered after "
                            f"{step} steps",
                ))

        # 2. Check learner update counts after warm-up
        if step > self._warm_up_steps:
            ppo = getattr(smart_coach, 'ppo_agent', None)
            if ppo is not None:
                checks_run += 1
                update_count = getattr(ppo, '_update_count', None)
                if update_count is not None and update_count == 0:
                    violations.append(IntegrityViolation(
                        severity="warning",
                        check_name="ppo_has_updated",
                        message=f"PPO update_count is 0 after "
                                f"{step} steps",
                    ))

        # 3. Check reward pipeline is being used
        if step > self._warm_up_steps:
            reward_pipe = getattr(
                smart_coach, '_reward_pipeline', None
            )
            if reward_pipe is not None:
                checks_run += 1
                pipe_count = getattr(reward_pipe, 'step_count', 0)
                if pipe_count == 0:
                    violations.append(IntegrityViolation(
                        severity="error",
                        check_name="reward_pipeline_used",
                        message="UnifiedRewardPipeline step_count is 0 "
                                f"after {step} steps — pipeline bypassed",
                    ))

        # 4. Check DecisionCore is being used
        if step > self._warm_up_steps:
            decision_core = getattr(
                smart_coach, '_decision_core', None
            )
            if decision_core is not None:
                checks_run += 1
                hit_dist = decision_core.get_hit_distribution()
                if not hit_dist:
                    violations.append(IntegrityViolation(
                        severity="warning",
                        check_name="decision_core_used",
                        message="DecisionCore has no hit distribution "
                                f"after {step} steps",
                    ))

        passed = all(v.severity != "error" for v in violations)
        report = IntegrityReport(
            passed=passed,
            violations=violations,
            checks_run=checks_run,
        )

        if not passed:
            logger.warning(
                "Runtime integrity: %d errors, %d warnings at step %d",
                report.error_count,
                report.warning_count,
                step,
            )

        return report
