"""
core/ops/ops_dashboard_panels.py — Phase 38.5: OPS Dashboard Data Providers

Provides structured data dictionaries for the LiveDashboard to render
OPS-specific panels.  These functions do NOT create Rich renderables
directly — they return plain dicts that the dashboard can format.

Panels:
  1. OPS Status     — Lockout, cooldown, confidence summary
  2. Engagement     — Metrics snapshot, stagnation, velocity
  3. Token Flex     — Current flex scale, tier hints, signals
  4. Domain Intel   — Tracked domains, vhosts, hosts entries
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from core.ops.command_lockout import CommandLockout
    from core.ops.domain_manager import DomainManager
    from core.ops.engagement_metrics import EngagementMetrics
    from core.ops.exploit_confidence import ExploitConfidenceTracker
    from core.ops.exploit_cooldown import ExploitCooldownManager
    from core.ops.token_flex import TokenFlexResult

logger = logging.getLogger("ariaska.ops.dashboard")


def ops_status_panel(
    lockout: Optional[Any] = None,
    confidence: Optional[Any] = None,
    cooldown: Optional[Any] = None,
    current_step: int = 0,
) -> Dict[str, Any]:
    """
    Generate OPS status panel data.

    Returns:
        Dict with "lockout", "confidence", "cooldown" sub-dicts.
    """
    data: Dict[str, Any] = {
        "title": "OPS Status",
        "lockout": {"locked_count": 0, "total_tracked": 0, "pressure": 0.0},
        "confidence": {"tracked": 0, "mean": 0.0, "low_count": 0},
        "cooldown": {"active": 0, "total": 0, "pressure": 0.0},
    }

    if lockout is not None:
        stats = lockout.get_stats()
        locked = stats.get("currently_locked", 0)
        total = stats.get("total_tracked", 0)
        data["lockout"] = {
            "locked_count": locked,
            "total_tracked": total,
            "pressure": round(locked / max(total, 1), 3),
        }

    if confidence is not None:
        ranked = confidence.get_ranked_exploits()
        if ranked:
            confs = [entry.current_confidence for entry in ranked]
            data["confidence"] = {
                "tracked": len(ranked),
                "mean": round(sum(confs) / len(confs), 3),
                "low_count": sum(1 for c in confs if c < 0.35),
            }

    if cooldown is not None:
        entries = cooldown._entries if hasattr(cooldown, "_entries") else {}
        total_tracked = len(entries)
        if total_tracked > 0:
            all_names = list(entries.keys())
            avail = cooldown.get_available_exploits(all_names, current_step)
            on_cd = max(0, total_tracked - len(avail))
        else:
            on_cd = 0
        data["cooldown"] = {
            "active": on_cd,
            "total": total_tracked,
            "pressure": round(on_cd / max(total_tracked, 1), 3),
        }

    return data


def engagement_panel(
    metrics: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Generate engagement metrics panel data.

    Returns:
        Dict with engagement progress, stagnation, velocity,
        discovery rate, exploit success rate, etc.
    """
    data: Dict[str, Any] = {
        "title": "Engagement Metrics",
        "total_steps": 0,
        "current_phase": "RECON",
        "highest_phase": "RECON",
        "phase_progress": 0.0,
        "total_discoveries": 0,
        "discovery_rate": 0.0,
        "stagnation_level": 0.0,
        "phase_velocity": 0.0,
        "token_efficiency": 0.0,
        "exploit_success_rate": 0.0,
        "flags_captured": 0,
        "shells_obtained": 0,
    }

    if metrics is not None:
        progress = metrics.get_progress()
        data.update({
            "total_steps": progress.get("total_steps", 0),
            "current_phase": progress.get("current_phase", "RECON"),
            "highest_phase": progress.get("highest_phase", "RECON"),
            "phase_progress": round(progress.get("phase_progress", 0.0), 3),
            "total_discoveries": progress.get("total_discoveries", 0),
            "discovery_rate": round(progress.get("discovery_rate", 0.0), 4),
            "stagnation_level": round(
                metrics.get_stagnation_level(), 3),
            "phase_velocity": round(metrics.get_phase_velocity(), 3),
            "token_efficiency": round(
                metrics.get_token_efficiency(), 3),
            "exploit_success_rate": round(
                progress.get("exploit_success_rate", 0.0), 3),
            "flags_captured": progress.get("flags_count", 0),
            "shells_obtained": progress.get("shells_obtained", 0),
        })

    return data


def token_flex_panel(
    flex_result: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Generate token flex panel data.

    Returns:
        Dict with scale, reason, tier hints.
    """
    data: Dict[str, Any] = {
        "title": "Token Flex",
        "scale": 1.0,
        "reason": "default",
        "tier_hints": {},
        "signals": {},
    }

    if flex_result is not None:
        data.update({
            "scale": round(flex_result.scale, 4),
            "reason": flex_result.reason,
            "tier_hints": {
                k: round(v, 3) for k, v in flex_result.tier_hints.items()
            },
            "signals": {
                k: round(v, 4) for k, v in flex_result.signals.items()
            },
        })

    return data


def domain_intel_panel(
    domain_manager: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Generate domain intelligence panel data.

    Returns:
        Dict with primary domain, confirmed domains, vhosts,
        hosts entries, and total domain count.
    """
    data: Dict[str, Any] = {
        "title": "Domain Intel",
        "primary": None,
        "confirmed_count": 0,
        "vhost_count": 0,
        "confirmed_domains": [],
        "vhosts": [],
        "hosts_entries": [],
    }

    if domain_manager is not None:
        ctx = domain_manager.get_context()
        confirmed = domain_manager.get_confirmed_domains()
        vhosts = domain_manager.get_vhosts()
        hosts = domain_manager.get_hosts_entries()

        data.update({
            "primary": ctx.get("primary_domain"),
            "confirmed_count": len(confirmed),
            "vhost_count": len(vhosts),
            "confirmed_domains": list(confirmed)[:20],  # cap for display
            "vhosts": list(vhosts)[:20],
            "hosts_entries": [
                f"{ip} {domain}" for ip, domain in hosts[:15]
            ],
        })

    return data


def all_ops_panels(
    lockout: Optional[Any] = None,
    confidence: Optional[Any] = None,
    cooldown: Optional[Any] = None,
    metrics: Optional[Any] = None,
    flex_result: Optional[Any] = None,
    domain_manager: Optional[Any] = None,
    current_step: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """
    Collect all OPS panel data in one call.

    Returns:
        Dict mapping panel name to panel data dict.
    """
    return {
        "ops_status": ops_status_panel(
            lockout=lockout,
            confidence=confidence,
            cooldown=cooldown,
            current_step=current_step,
        ),
        "engagement": engagement_panel(metrics=metrics),
        "token_flex": token_flex_panel(flex_result=flex_result),
        "domain_intel": domain_intel_panel(domain_manager=domain_manager),
    }
