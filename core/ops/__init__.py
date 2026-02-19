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
