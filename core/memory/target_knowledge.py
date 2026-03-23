#!/usr/bin/env python3
"""
core/memory/target_knowledge.py — Phase 58: Per-Target Persistent Learning Memory

Tracks what the system has learned about EACH specific target across episodes.
Unlike CampaignMemory (which is a single global store), TargetKnowledge is
keyed by target IP/hostname and remembers:

  - Which exploits worked/failed on THIS target
  - Which services were confirmed on THIS target
  - Optimal attack chains discovered for THIS target
  - Privesc vectors confirmed on THIS target
  - Credential pairs valid on THIS target

This enables the system to:
  1. Skip known-bad approaches on re-engagement
  2. Re-use confirmed attack chains (warm start)
  3. Prioritize hypotheses that match prior success on this target class

Persistence: JSON file per target in data/target_knowledge/

Architecture:
    TargetKnowledge
    ├── load(target_id) → load prior knowledge for this target
    ├── save() → persist current knowledge
    ├── record_exploit_result(template, success, reward) → learn from attempt
    ├── record_service(port, service, version) → confirm service
    ├── record_credential(user, pass, service) → confirm credential
    ├── record_privesc(method, success) → confirm privesc vector
    ├── get_best_chain() → return best known attack chain
    ├── get_failed_exploits() → return known-bad exploits to avoid
    ├── get_service_hypotheses() → return hypotheses from confirmed services
    └── merge_into_state(state_dict) → inject prior knowledge into state

Author: Filip Volf
Phase: 58 — Per-Target Persistent Learning
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("ariaska.target_knowledge")


@dataclass
class ExploitAttempt:
    """Record of an exploit attempt on this target."""
    template: str
    command: str = ""
    success: bool = False
    reward: float = 0.0
    attempt_count: int = 1
    last_episode: int = 0
    last_timestamp: float = field(default_factory=time.time)


@dataclass
class ConfirmedService:
    """A confirmed service on this target."""
    port: int
    service: str
    version: str = ""
    confirmed_count: int = 1
    exploits_tried: List[str] = field(default_factory=list)
    exploits_succeeded: List[str] = field(default_factory=list)


@dataclass
class ConfirmedPrivesc:
    """A confirmed privesc vector on this target."""
    method: str
    success: bool = False
    command: str = ""
    attempt_count: int = 1
    last_episode: int = 0


@dataclass
class TargetAttackChain:
    """A recorded full attack chain for this target."""
    steps: List[str]
    highest_phase: str = "RECON"
    total_reward: float = 0.0
    episode: int = 0
    timestamp: float = field(default_factory=time.time)


class TargetKnowledge:
    """
    Per-target persistent learning memory.

    Stores and retrieves target-specific knowledge across episodes.
    Enables warm starts and exploit prioritization based on prior experience.
    """

    def __init__(self, base_dir: str = "data/target_knowledge") -> None:
        self._base_dir = Path(base_dir)
        self._target_id: str = ""
        self._path: Optional[Path] = None

        # Knowledge stores
        self.services: Dict[int, ConfirmedService] = {}
        self.exploit_attempts: Dict[str, ExploitAttempt] = {}
        self.credentials: List[Dict[str, str]] = []
        self.privesc_vectors: Dict[str, ConfirmedPrivesc] = {}
        self.attack_chains: List[TargetAttackChain] = []
        self.os_info: str = ""
        self.kernel_version: str = ""

        # Metadata
        self.total_engagements: int = 0
        self.best_phase: str = "RECON"
        self.last_engagement: float = 0.0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self, target_id: str) -> bool:
        """Load prior knowledge for a specific target. Returns True if loaded."""
        self._target_id = self._sanitize_target_id(target_id)
        self._path = self._base_dir / f"{self._target_id}.json"

        if not self._path.exists():
            logger.info(f"[TARGET-KB] No prior knowledge for {target_id}")
            return False

        try:
            with open(self._path) as f:
                data = json.load(f)

            self.total_engagements = data.get("total_engagements", 0)
            self.best_phase = data.get("best_phase", "RECON")
            self.last_engagement = data.get("last_engagement", 0.0)
            self.os_info = data.get("os_info", "")
            self.kernel_version = data.get("kernel_version", "")

            for port_str, sdata in data.get("services", {}).items():
                port = int(port_str)
                self.services[port] = ConfirmedService(**{
                    k: v for k, v in sdata.items()
                    if k in ConfirmedService.__dataclass_fields__
                })

            for eid, edata in data.get("exploit_attempts", {}).items():
                self.exploit_attempts[eid] = ExploitAttempt(**{
                    k: v for k, v in edata.items()
                    if k in ExploitAttempt.__dataclass_fields__
                })

            self.credentials = data.get("credentials", [])

            for pid, pdata in data.get("privesc_vectors", {}).items():
                self.privesc_vectors[pid] = ConfirmedPrivesc(**{
                    k: v for k, v in pdata.items()
                    if k in ConfirmedPrivesc.__dataclass_fields__
                })

            for cdata in data.get("attack_chains", []):
                self.attack_chains.append(TargetAttackChain(**{
                    k: v for k, v in cdata.items()
                    if k in TargetAttackChain.__dataclass_fields__
                }))

            logger.info(
                f"[TARGET-KB] Loaded {target_id}: {len(self.services)} services, "
                f"{len(self.exploit_attempts)} exploits, {len(self.credentials)} creds, "
                f"best_phase={self.best_phase}, engagements={self.total_engagements}"
            )
            return True

        except Exception as e:
            logger.warning(f"[TARGET-KB] Failed to load {target_id}: {e}")
            return False

    def save(self) -> None:
        """Save current target knowledge to JSON."""
        if not self._path:
            return

        self._base_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": "1.0",
            "target_id": self._target_id,
            "total_engagements": self.total_engagements,
            "best_phase": self.best_phase,
            "last_engagement": time.time(),
            "os_info": self.os_info,
            "kernel_version": self.kernel_version,
            "services": {
                str(k): asdict(v) for k, v in self.services.items()
            },
            "exploit_attempts": {
                k: asdict(v) for k, v in self.exploit_attempts.items()
            },
            "credentials": self.credentials[-50:],
            "privesc_vectors": {
                k: asdict(v) for k, v in self.privesc_vectors.items()
            },
            "attack_chains": [
                asdict(c) for c in self.attack_chains[-10:]
            ],
        }

        with open(self._path, "w") as f:
            json.dump(payload, f, indent=2, default=str)

        logger.info(
            f"[TARGET-KB] Saved {self._target_id}: "
            f"{len(self.services)} services, {len(self.exploit_attempts)} exploits"
        )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_exploit_result(
        self,
        template: str,
        success: bool,
        reward: float = 0.0,
        command: str = "",
        episode: int = 0,
    ) -> None:
        """Record an exploit attempt result on this target."""
        if template in self.exploit_attempts:
            ea = self.exploit_attempts[template]
            ea.attempt_count += 1
            ea.last_episode = episode
            ea.last_timestamp = time.time()
            if success:
                ea.success = True
                ea.reward = max(ea.reward, reward)
        else:
            self.exploit_attempts[template] = ExploitAttempt(
                template=template,
                command=command[:200],
                success=success,
                reward=reward,
                last_episode=episode,
            )

        # Update service records
        for svc in self.services.values():
            if template not in svc.exploits_tried:
                svc.exploits_tried.append(template)
            if success and template not in svc.exploits_succeeded:
                svc.exploits_succeeded.append(template)

    def record_service(
        self,
        port: int,
        service: str,
        version: str = "",
    ) -> None:
        """Record a confirmed service on this target."""
        if port in self.services:
            cs = self.services[port]
            cs.confirmed_count += 1
            if version and not cs.version:
                cs.version = version
        else:
            self.services[port] = ConfirmedService(
                port=port,
                service=service,
                version=version,
            )

    def record_credential(
        self,
        username: str,
        password: str,
        service: str = "",
    ) -> None:
        """Record a confirmed credential pair."""
        cred = {"username": username, "password": password, "service": service}
        if not any(
            c.get("username") == username and c.get("password") == password
            for c in self.credentials
        ):
            self.credentials.append(cred)

    def record_privesc(
        self,
        method: str,
        success: bool,
        command: str = "",
        episode: int = 0,
    ) -> None:
        """Record a privesc attempt result."""
        if method in self.privesc_vectors:
            pv = self.privesc_vectors[method]
            pv.attempt_count += 1
            pv.last_episode = episode
            if success:
                pv.success = True
        else:
            self.privesc_vectors[method] = ConfirmedPrivesc(
                method=method,
                success=success,
                command=command[:200],
                last_episode=episode,
            )

    def record_attack_chain(
        self,
        steps: List[str],
        highest_phase: str,
        total_reward: float = 0.0,
        episode: int = 0,
    ) -> None:
        """Record a completed attack chain."""
        self.attack_chains.append(TargetAttackChain(
            steps=steps[:50],
            highest_phase=highest_phase,
            total_reward=total_reward,
            episode=episode,
        ))
        # Keep best 10 by phase reached
        phase_order = [
            "RECON", "ENUMERATION", "EXPLOITATION",
            "PRIVILEGE_ESCALATION", "LATERAL_MOVEMENT",
            "POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT",
        ]
        self.attack_chains.sort(
            key=lambda c: phase_order.index(c.highest_phase)
            if c.highest_phase in phase_order else 0,
            reverse=True,
        )
        self.attack_chains = self.attack_chains[:10]

        # Update best phase
        if highest_phase in phase_order:
            cur = phase_order.index(self.best_phase) if self.best_phase in phase_order else 0
            new = phase_order.index(highest_phase)
            if new > cur:
                self.best_phase = highest_phase

    def record_os_info(self, os_info: str, kernel_version: str = "") -> None:
        """Record OS information for this target."""
        if os_info:
            self.os_info = os_info
        if kernel_version:
            self.kernel_version = kernel_version

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_failed_exploits(self) -> Set[str]:
        """Return exploit templates that have consistently failed on this target."""
        return {
            t for t, ea in self.exploit_attempts.items()
            if not ea.success and ea.attempt_count >= 2
        }

    def get_successful_exploits(self) -> Set[str]:
        """Return exploit templates that succeeded on this target."""
        return {
            t for t, ea in self.exploit_attempts.items()
            if ea.success
        }

    def get_best_chain(self) -> Optional[TargetAttackChain]:
        """Return the best known attack chain for this target."""
        if not self.attack_chains:
            return None
        return self.attack_chains[0]

    def get_service_list(self) -> List[Dict[str, str]]:
        """Return confirmed services as dicts for ExploitReasoner."""
        return [
            {
                "service": svc.service,
                "version": svc.version,
                "port": str(svc.port),
            }
            for svc in self.services.values()
        ]

    def get_credential_list(self) -> List[Dict[str, str]]:
        """Return confirmed credentials for credential reuse reasoning."""
        return list(self.credentials)

    def get_untried_exploits_for_service(self, port: int) -> List[str]:
        """Return exploits that succeeded elsewhere but haven't been tried on this port."""
        if port not in self.services:
            return []
        svc = self.services[port]
        return [
            t for t, ea in self.exploit_attempts.items()
            if ea.success and t not in svc.exploits_tried
        ]

    def get_hypothesis_boost(self, template: str) -> float:
        """
        Return confidence boost for a hypothesis based on prior knowledge.

        Returns:
            Positive float if known-good, negative if known-bad, 0 if unknown.
        """
        if template not in self.exploit_attempts:
            return 0.0
        ea = self.exploit_attempts[template]
        if ea.success:
            return min(0.30, 0.10 * ea.attempt_count)
        else:
            return max(-0.30, -0.10 * ea.attempt_count)

    def merge_into_state(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inject prior target knowledge into a state dict for the state encoder.

        Adds confirmed ports, services, credentials to the state so agents
        start with prior knowledge instead of discovering from scratch.
        """
        if self.services:
            state_dict.setdefault("ports_discovered", True)
            known_ports = {svc.port for svc in self.services.values()}
            state_dict["known_ports"] = known_ports

        if self.credentials:
            state_dict["has_prior_creds"] = True
            state_dict["prior_cred_count"] = len(self.credentials)

        if any(ea.success for ea in self.exploit_attempts.values()):
            state_dict["has_known_exploits"] = True

        if self.best_phase != "RECON":
            state_dict["prior_best_phase"] = self.best_phase

        if self.os_info:
            state_dict["os_info"] = self.os_info
        if self.kernel_version:
            state_dict["kernel_version"] = self.kernel_version

        return state_dict

    def reset_episode(self) -> None:
        """Mark a new engagement episode (don't clear knowledge)."""
        self.total_engagements += 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_target_id(target_id: str) -> str:
        """Sanitize target ID for use as filename. Only allow safe chars."""
        safe = ""
        for c in target_id:
            if c.isalnum() or c in (".", "-", "_"):
                safe += c
            else:
                safe += "_"
        return safe or "unknown"

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics."""
        return {
            "target": self._target_id,
            "services": len(self.services),
            "exploit_attempts": len(self.exploit_attempts),
            "successful_exploits": len(self.get_successful_exploits()),
            "failed_exploits": len(self.get_failed_exploits()),
            "credentials": len(self.credentials),
            "privesc_vectors": len(self.privesc_vectors),
            "attack_chains": len(self.attack_chains),
            "best_phase": self.best_phase,
            "total_engagements": self.total_engagements,
        }
