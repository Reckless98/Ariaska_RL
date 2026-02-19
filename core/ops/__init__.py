# core/ops/__init__.py — OPS Authority Package
# Phase A: Autonomous environment preparation, sudo handling,
# tool installation, hosts management, execution classification.
#
# ScoutAgent → OPS: recon + discovery + env prep + tool verification + hosts
# ShadowAgent → OPS: system integrity + phase verify + state audit + guardrails

__all__ = [
    "SudoHandler",
    "HostsManager",
    "ToolInstaller",
    "ExecutionClassifier",
    "ExecutionClass",
    "DiscoveryTrustEngine",
    "VerificationLevel",
    "PhaseInvariantChecker",
    "ShellValidator",
    "DomainManager",
    "CommandLockout",
    "ExploitConfidenceTracker",
    "ExploitCooldownManager",
    "EngagementMetrics",
    "TokenFlexEngine",
    "inject_ops_features",
    "collect_ops_signals",
    "all_ops_panels",
    "OpsHub",
    "OpsHubConfig",
    # Phase 39.1: Orion Rethink
    "OrionRethinkEngine",
    "StallSignals",
    "OrionRethinkPlan",
    # Phase 39.2: Trust Weights
    "TrustWeightEngine",
    "TrustInfluenceResult",
    # Phase 39.4: Debug Trace
    "DebugTracer",
    "DebugTraceEntry",
]


def _lazy_sudo_handler():
    from core.ops.sudo_handler import SudoHandler
    return SudoHandler


def _lazy_hosts_manager():
    from core.ops.hosts_manager import HostsManager
    return HostsManager


def _lazy_tool_installer():
    from core.ops.tool_installer import ToolInstaller
    return ToolInstaller


def _lazy_execution_classifier():
    from core.ops.execution_classifier import ExecutionClassifier, ExecutionClass
    return ExecutionClassifier, ExecutionClass


def _lazy_phase_invariant_checker():
    from core.ops.phase_invariants import PhaseInvariantChecker
    return PhaseInvariantChecker


def _lazy_shell_validator():
    from core.ops.shell_validator import ShellValidator
    return ShellValidator


def _lazy_domain_manager():
    from core.ops.domain_manager import DomainManager
    return DomainManager


def _lazy_command_lockout():
    from core.ops.command_lockout import CommandLockout
    return CommandLockout


def _lazy_exploit_confidence():
    from core.ops.exploit_confidence import ExploitConfidenceTracker
    return ExploitConfidenceTracker


def _lazy_exploit_cooldown():
    from core.ops.exploit_cooldown import ExploitCooldownManager
    return ExploitCooldownManager


def _lazy_engagement_metrics():
    from core.ops.engagement_metrics import EngagementMetrics
    return EngagementMetrics


def _lazy_token_flex():
    from core.ops.token_flex import TokenFlexEngine
    return TokenFlexEngine


def _lazy_ops_state_encoder():
    from core.ops.ops_state_encoder import inject_ops_features, collect_ops_signals
    return inject_ops_features, collect_ops_signals


def _lazy_ops_dashboard():
    from core.ops.ops_dashboard_panels import all_ops_panels
    return all_ops_panels


def _lazy_ops_hub():
    from core.ops.ops_hub import OpsHub, OpsHubConfig
    return OpsHub, OpsHubConfig


def _lazy_orion_rethink():
    from core.ops.orion_rethink import (
        OrionRethinkEngine,
        StallSignals,
        OrionRethinkPlan,
    )
    return OrionRethinkEngine, StallSignals, OrionRethinkPlan


def _lazy_trust_weights():
    from core.ops.trust_weights import TrustWeightEngine, TrustInfluenceResult
    return TrustWeightEngine, TrustInfluenceResult


def _lazy_debug_trace():
    from core.ops.debug_trace import DebugTracer, DebugTraceEntry
    return DebugTracer, DebugTraceEntry
