"""
core/ops/ops_hub.py — Phase 38.6: OPS Integration Hub

Central integration point for all OPS subsystems.  SmartOrchestrator
instantiates ONE OpsHub and passes it to SmartCoach instances.

Lifecycle:
  1.  SmartOrchestrator.__init__  → OpsHub(gpt_manager, config)
  2.  Per-step                   → hub.on_step(step_ctx)
  3.  Per-agent decision         → hub.pre_decide / hub.post_decide
  4.  Per-episode end            → hub.on_episode_end(metrics)
  5.  State encoding             → hub.enrich_state(vec)
  6.  Dashboard                  → hub.get_dashboard_data()

Dependencies (all lazy-imported, all from core.ops):
  - SudoHandler, HostsManager, ToolInstaller  (Phase A)
  - ExecutionClassifier                        (Phase A)
  - DiscoveryTrustEngine                       (Phase B)
  - PhaseInvariantChecker, ShellValidator       (Phase C)
  - DomainManager                              (Phase C)
  - CommandLockout, ExploitConfidenceTracker    (Phase D)
  - ExploitCooldownManager                     (Phase D)
  - EngagementMetrics, TokenFlexEngine          (Phase E)
  - inject_ops_features, collect_ops_signals    (Phase F)
  - all_ops_panels                             (Phase F)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger("ariaska.ops.hub")


@dataclass
class OpsHubConfig:
    """Configuration for OpsHub."""
    max_steps: int = 500
    target_ip: str = ""
    primary_domain: str = ""
    strict_phase: bool = True
    enable_lockout: bool = True
    enable_confidence: bool = True
    enable_cooldown: bool = True
    enable_trust: bool = True
    enable_flex: bool = True
    enable_metrics: bool = True
    enable_dashboard: bool = True


class OpsHub:
    """
    Central OPS integration hub.

    Owns all OPS module instances and provides a clean API for
    SmartOrchestrator and SmartCoach to consume.

    Usage:
        hub = OpsHub(config=OpsHubConfig(max_steps=500, target_ip="10.10.11.23"))
        hub.setup(primary_domain="permx.htb")

        # Per step
        hub.on_step_start(step=1, phase="RECON")

        # Pre-decision check
        available = hub.filter_available_commands(
            candidates=["nmap_scan", "dirb_scan"],
            current_step=5,
        )

        # Post-decision recording
        hub.record_command_result(
            template_name="nmap_scan",
            command="nmap -sV 10.10.11.23",
            output="22/tcp open ssh...",
            success=True,
            discoveries=3,
            tokens=150,
            step=1,
        )

        # Episode end
        hub.on_episode_end(episode=0)

        # State enrichment
        vec = np.zeros(512, dtype=np.float32)
        hub.enrich_state(vec, current_step=5)

        # Dashboard data
        panels = hub.get_dashboard_data()
    """

    def __init__(self, config: Optional[OpsHubConfig] = None) -> None:
        self._config = config or OpsHubConfig()
        self._initialised = False
        self._current_step: int = 0
        self._current_phase: str = "RECON"

        # Lazy-init all modules
        self._lockout: Optional[Any] = None
        self._confidence: Optional[Any] = None
        self._cooldown: Optional[Any] = None
        self._trust: Optional[Any] = None
        self._phase_checker: Optional[Any] = None
        self._shell_validator: Optional[Any] = None
        self._domain_manager: Optional[Any] = None
        self._metrics: Optional[Any] = None
        self._flex_engine: Optional[Any] = None
        self._last_flex_result: Optional[Any] = None

        self._init_modules()
        logger.info("OpsHub initialised with config: %s", self._config)

    def _init_modules(self) -> None:
        """Initialise all OPS modules (lazy imports)."""
        if self._config.enable_lockout:
            from core.ops.command_lockout import CommandLockout
            self._lockout = CommandLockout()

        if self._config.enable_confidence:
            from core.ops.exploit_confidence import ExploitConfidenceTracker
            self._confidence = ExploitConfidenceTracker()

        if self._config.enable_cooldown:
            from core.ops.exploit_cooldown import ExploitCooldownManager
            self._cooldown = ExploitCooldownManager()

        if self._config.enable_trust:
            from core.ops.discovery_trust import DiscoveryTrustEngine
            self._trust = DiscoveryTrustEngine()

        from core.ops.phase_invariants import PhaseInvariantChecker
        self._phase_checker = PhaseInvariantChecker(
            strict=self._config.strict_phase,
        )

        from core.ops.shell_validator import ShellValidator
        self._shell_validator = ShellValidator()

        from core.ops.domain_manager import DomainManager
        self._domain_manager = DomainManager()

        if self._config.enable_metrics:
            from core.ops.engagement_metrics import EngagementMetrics
            self._metrics = EngagementMetrics()

        if self._config.enable_flex:
            from core.ops.token_flex import TokenFlexEngine
            self._flex_engine = TokenFlexEngine(
                max_steps=self._config.max_steps,
            )

        self._initialised = True

    def setup(self, primary_domain: str = "", target_ip: str = "") -> None:
        """
        One-time setup with engagement-specific parameters.

        Called after construction when target info is known.
        """
        ip = target_ip or self._config.target_ip
        domain = primary_domain or self._config.primary_domain

        if domain and self._domain_manager is not None:
            self._domain_manager.set_primary(domain, ip=ip)

    # ── Per-Step Hooks ───────────────────────────────────────────────────

    def on_step_start(
        self,
        step: int,
        phase: str = "",
    ) -> None:
        """Called at the beginning of each step."""
        self._current_step = step
        if phase:
            self._current_phase = phase.upper()

    def on_step_end(
        self,
        step: int,
        phase: str = "",
        discoveries: int = 0,
        command: str = "",
        tokens: int = 0,
        shell_obtained: bool = False,
    ) -> None:
        """Called at the end of each step to update metrics."""
        if self._metrics is not None:
            self._metrics.record_step(
                step=step,
                phase=phase or self._current_phase,
                discoveries=discoveries,
                command=command,
                tokens=tokens,
                shell_obtained=shell_obtained,
            )

        # Recompute flex
        if self._flex_engine is not None and self._metrics is not None:
            progress = self._metrics.get_progress()
            self._last_flex_result = self._flex_engine.compute(
                phase=self._current_phase,
                step=step,
                stagnation_level=self._metrics.get_stagnation_level(),
                flags_captured=progress.get("flags_count", 0),
                shells_obtained=progress.get("shells_obtained", 0),
                exploit_success_rate=progress.get("exploit_success_rate", 0.0),
                discovery_rate=progress.get("discovery_rate", 0.0),
            )

    # ── Pre-Decision Hooks ───────────────────────────────────────────────

    def filter_available_commands(
        self,
        candidates: List[str],
        current_step: int,
    ) -> List[str]:
        """
        Filter candidate template names by lockout and cooldown.

        Returns only templates that are not locked out or on cooldown.
        """
        available = list(candidates)

        if self._lockout is not None:
            available = [
                t for t in available
                if not self._lockout.is_locked(t, current_step)
            ]

        if self._cooldown is not None:
            available = self._cooldown.get_available_exploits(
                available, current_step,
            )

        return available

    def get_exploit_confidence(self, template_name: str) -> float:
        """Get confidence for an exploit template. Returns 0.0 if unknown."""
        if self._confidence is not None:
            return self._confidence.get_confidence(template_name)
        return 0.0

    def is_low_confidence(self, template_name: str) -> bool:
        """Check if an exploit is below confidence threshold."""
        if self._confidence is not None:
            return self._confidence.is_low_confidence(template_name)
        return False

    def validate_phase_transition(
        self,
        current_phase: str,
        requested_phase: str,
        state_flags: Dict[str, bool],
        discovery_board: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Validate a phase transition using PhaseInvariantChecker.

        Returns dict with 'valid', 'details', 'recommended_phase'.
        """
        if self._phase_checker is not None:
            result = self._phase_checker.validate_transition(
                current_phase=current_phase,
                requested_phase=requested_phase,
                state_flags=state_flags,
                discovery_board=discovery_board or {},
            )
            return {
                "valid": result.valid,
                "details": result.details,
                "recommended_phase": result.recommended_phase,
            }
        return {"valid": True, "details": "", "recommended_phase": requested_phase}

    def validate_shell(
        self,
        command: str,
        output: str,
        target_ip: str = "",
        domain: str = "",
    ) -> Dict[str, Any]:
        """
        Validate shell evidence from command output.

        Returns dict with validation result.
        """
        if self._shell_validator is not None:
            result = self._shell_validator.validate(
                command=command,
                output=output,
                target_ip=target_ip or self._config.target_ip,
                domain=domain,
            )
            return {
                "is_valid_shell": result.is_valid_shell,
                "is_root_shell": result.is_root_shell,
                "confidence": result.confidence,
                "evidence": result.evidence,
                "rejection_reason": result.rejection_reason,
            }
        return {
            "is_valid_shell": False,
            "is_root_shell": False,
            "confidence": 0.0,
            "evidence": "",
            "rejection_reason": "shell_validator_disabled",
        }

    def get_token_flex_scale(self) -> float:
        """Get current token flex scale [0.5, 1.5]."""
        if self._last_flex_result is not None:
            return self._last_flex_result.scale
        return 1.0

    def get_token_flex_tier_hints(self) -> Dict[str, float]:
        """Get per-tier adjustment hints from token flex."""
        if self._last_flex_result is not None:
            return dict(self._last_flex_result.tier_hints)
        return {}

    # ── Post-Decision Hooks ──────────────────────────────────────────────

    def record_command_result(
        self,
        template_name: str,
        command: str = "",
        output: str = "",
        success: bool = False,
        discoveries: int = 0,
        tokens: int = 0,
        step: int = 0,
        is_exploit: bool = False,
    ) -> None:
        """Record the result of a command execution."""
        step = step or self._current_step

        if self._lockout is not None:
            self._lockout.record_result(template_name, success=success, step=step)

        if is_exploit:
            if self._cooldown is not None:
                self._cooldown.record_attempt(
                    template_name, step=step, success=success,
                )
            if self._confidence is not None:
                self._confidence.record_attempt(
                    template_name, success=success, step=step,
                )
            if self._metrics is not None:
                self._metrics.record_exploit_attempt(success=success)

        # Domain extraction from output
        if output and self._domain_manager is not None:
            self._domain_manager.extract_domains_from_output(output)

    def register_exploit(
        self,
        template_name: str,
        service: str = "",
        base_confidence: float = 0.5,
    ) -> None:
        """Register an exploit for confidence tracking."""
        if self._confidence is not None:
            self._confidence.register_exploit(
                template_name, service=service,
                base_confidence=base_confidence,
            )

    def add_exploit_evidence(
        self,
        template_name: str,
        evidence: str,
    ) -> None:
        """Add evidence supporting an exploit."""
        if self._confidence is not None:
            self._confidence.add_evidence(template_name, evidence)

    def record_flag(self, flag_type: str, step: int) -> None:
        """Record a flag capture."""
        if self._metrics is not None:
            self._metrics.record_flag(flag_type, step)

    # ── Episode Lifecycle ────────────────────────────────────────────────

    def on_episode_end(self, episode: int = 0) -> Dict[str, Any]:
        """
        Called at the end of each episode.

        Returns engagement progress dict.
        """
        if self._metrics is not None:
            self._metrics.record_episode_end(episode)
            return self._metrics.get_progress()
        return {}

    def reset(self) -> None:
        """Reset all OPS modules for a new engagement."""
        self._current_step = 0
        self._current_phase = "RECON"
        self._last_flex_result = None

        if self._lockout is not None:
            self._lockout.reset()
        if self._confidence is not None:
            self._confidence.reset()
        if self._cooldown is not None:
            self._cooldown.reset()
        if self._metrics is not None:
            self._metrics.reset()
        if self._domain_manager is not None:
            self._domain_manager = None
            from core.ops.domain_manager import DomainManager
            self._domain_manager = DomainManager()

        logger.info("OpsHub reset for new engagement")

    # ── State Enrichment ─────────────────────────────────────────────────

    def enrich_state(
        self,
        vec: np.ndarray,
        current_step: int = 0,
        budget_stats: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Inject OPS features into the 512-dim state vector.

        Fills dims [237-269] with OPS intelligence signals.
        Modifies vec in-place and returns it.
        """
        from core.ops.ops_state_encoder import collect_ops_signals, inject_ops_features

        signals = collect_ops_signals(
            lockout=self._lockout,
            confidence=self._confidence,
            cooldown=self._cooldown,
            metrics=self._metrics,
            flex_result=self._last_flex_result,
            domain_manager=self._domain_manager,
            budget_stats=budget_stats,
            current_step=current_step or self._current_step,
        )

        return inject_ops_features(vec, **signals)

    # ── Dashboard ────────────────────────────────────────────────────────

    def get_dashboard_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all OPS dashboard panel data.

        Returns dict mapping panel name -> panel data.
        """
        from core.ops.ops_dashboard_panels import all_ops_panels

        return all_ops_panels(
            lockout=self._lockout,
            confidence=self._confidence,
            cooldown=self._cooldown,
            metrics=self._metrics,
            flex_result=self._last_flex_result,
            domain_manager=self._domain_manager,
            current_step=self._current_step,
        )

    # ── Accessors ────────────────────────────────────────────────────────

    @property
    def lockout(self) -> Optional[Any]:
        return self._lockout

    @property
    def confidence(self) -> Optional[Any]:
        return self._confidence

    @property
    def cooldown(self) -> Optional[Any]:
        return self._cooldown

    @property
    def trust(self) -> Optional[Any]:
        return self._trust

    @property
    def phase_checker(self) -> Optional[Any]:
        return self._phase_checker

    @property
    def shell_validator(self) -> Optional[Any]:
        return self._shell_validator

    @property
    def domain_manager(self) -> Optional[Any]:
        return self._domain_manager

    @property
    def metrics(self) -> Optional[Any]:
        return self._metrics

    @property
    def flex_engine(self) -> Optional[Any]:
        return self._flex_engine

    @property
    def config(self) -> OpsHubConfig:
        return self._config

    def get_engagement_progress(self) -> Dict[str, Any]:
        """Get current engagement progress from metrics."""
        if self._metrics is not None:
            return self._metrics.get_progress()
        return {}

    def get_stagnation_level(self) -> float:
        """Get engagement stagnation [0.0, 1.0]."""
        if self._metrics is not None:
            return self._metrics.get_stagnation_level()
        return 0.0
