"""Orchestration package — SmartOrchestrator + extracted submodules.

Phase 41: Architecture hardening.
Backward compatibility:
    from core.orchestration.smart_orchestrator import SmartOrchestrator  # still works
    from core.orchestration import SmartOrchestrator                     # also works
"""
from __future__ import annotations

# Re-export for backward compat — lazy to avoid circular deps
def __getattr__(name: str):
    if name in ("SmartOrchestrator", "SmartOrchestratorConfig"):
        from core.orchestration.smart_orchestrator import SmartOrchestrator, SmartOrchestratorConfig
        g = globals()
        g["SmartOrchestrator"] = SmartOrchestrator
        g["SmartOrchestratorConfig"] = SmartOrchestratorConfig
        return g[name]
    if name in ("Orchestrator", "OrchestratorConfig"):
        from core.orchestration.orchestrator import Orchestrator, OrchestratorConfig
        g = globals()
        g["Orchestrator"] = Orchestrator
        g["OrchestratorConfig"] = OrchestratorConfig
        return g[name]
    raise AttributeError(f"module 'core.orchestration' has no attribute {name!r}")

__all__ = ["SmartOrchestrator", "SmartOrchestratorConfig", "Orchestrator", "OrchestratorConfig"]
