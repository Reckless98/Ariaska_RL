#!/usr/bin/env python3
"""
core/memory/campaign_memory.py — ARIASKA Cross-Episode Campaign Memory v1.0

Persistent memory that carries confirmed high-value discoveries across episodes.
Avoids re-discovery of known facts, enabling agents to start smarter each episode.

Persisted as JSON. Auto-loaded at training start, auto-saved at episode end.

Stored facts:
  - Confirmed open ports (port, service, version)
  - Valid credentials (user:pass pairs)
  - Confirmed vulnerabilities (CVE, type, target)
  - Shell access history (type, how obtained)
  - Failed approaches (to avoid repeating)
  - Best attack chains (ordered steps that reached EXFIL)

Usage:
    campaign = CampaignMemory("campaign_state.json")
    campaign.load()
    
    # At episode start — inject into state encoder
    prior = campaign.get_prior_knowledge()
    
    # At episode end — record confirmed discoveries
    campaign.record_episode(episode_num, discoveries, highest_phase, best_chain)
    campaign.save()
"""

import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("ariaska.campaign_memory")


@dataclass
class ConfirmedPort:
    """A confirmed open port from a previous episode."""
    port: int
    service: str = ""
    version: str = ""
    first_seen_episode: int = 0
    last_seen_episode: int = 0
    confirmed_count: int = 1


@dataclass
class ConfirmedCredential:
    """A confirmed valid credential pair."""
    username: str
    password: str
    service: str = ""
    target: str = ""
    first_seen_episode: int = 0
    confirmed_count: int = 1


@dataclass
class ConfirmedVuln:
    """A confirmed vulnerability."""
    vuln_id: str  # CVE or description
    vuln_type: str = ""  # "backdoor", "sqli", "rce", etc.
    target_service: str = ""
    exploit_path: str = ""  # e.g. "exploit/unix/ftp/vsftpd_234_backdoor"
    first_seen_episode: int = 0
    confirmed_count: int = 1


@dataclass
class AttackChain:
    """A recorded attack chain that reached a high phase."""
    episode: int
    highest_phase: str
    steps: List[str]  # Ordered command template names
    total_reward: float = 0.0
    discoveries_count: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class FailedApproach:
    """An approach that consistently fails — avoid repeating."""
    command_template: str
    phase: str
    failure_count: int = 1
    last_failure_episode: int = 0
    reason: str = ""


class CampaignMemory:
    """
    Persistent cross-episode campaign memory.
    
    Carries confirmed knowledge between episodes so agents
    don't waste steps re-discovering known ports/services.
    """

    def __init__(self, path: str = "data/campaign_state.json"):
        self._path = Path(path)
        
        # Core knowledge stores
        self.ports: Dict[int, ConfirmedPort] = {}
        self.credentials: List[ConfirmedCredential] = []
        self.vulns: Dict[str, ConfirmedVuln] = {}
        self.shells: List[Dict[str, Any]] = []
        self.attack_chains: List[AttackChain] = []
        self.failed_approaches: Dict[str, FailedApproach] = {}
        
        # Metadata
        self.total_episodes: int = 0
        self.best_phase_reached: str = "RECON"
        self.last_updated: float = 0.0
        self.target_ip: str = ""

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Save campaign memory to JSON."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        
        payload = {
            "version": "1.0",
            "total_episodes": self.total_episodes,
            "best_phase_reached": self.best_phase_reached,
            "last_updated": time.time(),
            "target_ip": self.target_ip,
            "ports": {
                str(k): asdict(v) for k, v in self.ports.items()
            },
            "credentials": [asdict(c) for c in self.credentials],
            "vulns": {k: asdict(v) for k, v in self.vulns.items()},
            "shells": self.shells[-20:],  # Keep last 20
            "attack_chains": [asdict(c) for c in self.attack_chains[-10:]],  # Keep best 10
            "failed_approaches": {
                k: asdict(v) for k, v in self.failed_approaches.items()
            },
        }
        
        with open(self._path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        
        logger.info(
            f"[CAMPAIGN] Saved: {len(self.ports)} ports, "
            f"{len(self.credentials)} creds, {len(self.vulns)} vulns, "
            f"{len(self.attack_chains)} chains → {self._path}"
        )

    def load(self) -> bool:
        """Load campaign memory from JSON. Returns True if loaded."""
        if not self._path.exists():
            logger.info(f"[CAMPAIGN] No prior campaign at {self._path}, starting fresh")
            return False
        
        try:
            with open(self._path) as f:
                data = json.load(f)
            
            self.total_episodes = data.get("total_episodes", 0)
            self.best_phase_reached = data.get("best_phase_reached", "RECON")
            self.last_updated = data.get("last_updated", 0.0)
            self.target_ip = data.get("target_ip", "")
            
            # Reconstruct ports
            for port_str, pdata in data.get("ports", {}).items():
                port = int(port_str)
                self.ports[port] = ConfirmedPort(**{
                    k: v for k, v in pdata.items()
                    if k in ConfirmedPort.__dataclass_fields__
                })
            
            # Reconstruct credentials
            for cdata in data.get("credentials", []):
                self.credentials.append(ConfirmedCredential(**{
                    k: v for k, v in cdata.items()
                    if k in ConfirmedCredential.__dataclass_fields__
                }))
            
            # Reconstruct vulns
            for vid, vdata in data.get("vulns", {}).items():
                self.vulns[vid] = ConfirmedVuln(**{
                    k: v for k, v in vdata.items()
                    if k in ConfirmedVuln.__dataclass_fields__
                })
            
            # Shells and chains
            self.shells = data.get("shells", [])
            for cdata in data.get("attack_chains", []):
                self.attack_chains.append(AttackChain(**{
                    k: v for k, v in cdata.items()
                    if k in AttackChain.__dataclass_fields__
                }))
            
            # Failed approaches
            for fid, fdata in data.get("failed_approaches", {}).items():
                self.failed_approaches[fid] = FailedApproach(**{
                    k: v for k, v in fdata.items()
                    if k in FailedApproach.__dataclass_fields__
                })
            
            logger.info(
                f"[CAMPAIGN] Loaded: {len(self.ports)} ports, "
                f"{len(self.credentials)} creds, {len(self.vulns)} vulns, "
                f"best_phase={self.best_phase_reached}, "
                f"episodes_so_far={self.total_episodes}"
            )
            return True
            
        except Exception as e:
            logger.warning(f"[CAMPAIGN] Failed to load {self._path}: {e}")
            return False

    # ------------------------------------------------------------------
    # Recording discoveries from an episode
    # ------------------------------------------------------------------

    def record_episode(
        self,
        episode_num: int,
        discoveries: Dict[str, Any],
        highest_phase: str,
        command_chain: Optional[List[str]] = None,
        total_reward: float = 0.0,
    ) -> None:
        """
        Record discoveries from a completed episode.
        
        Args:
            episode_num: Episode number
            discoveries: Dict of {type: values} from the episode
            highest_phase: Highest phase reached
            command_chain: Ordered list of command template names used
            total_reward: Total episode reward
        """
        self.total_episodes = max(self.total_episodes, episode_num + 1)
        
        # Update best phase
        phase_order = [
            "RECON", "ENUMERATION", "EXPLOITATION",
            "PRIVILEGE_ESCALATION", "LATERAL_MOVEMENT",
            "POST_EXPLOITATION", "EXFILTRATION",
        ]
        if highest_phase in phase_order:
            current_idx = phase_order.index(self.best_phase_reached) if self.best_phase_reached in phase_order else 0
            new_idx = phase_order.index(highest_phase)
            if new_idx > current_idx:
                self.best_phase_reached = highest_phase
        
        # Record ports
        for port in discoveries.get("open_port", []):
            port_int = int(port) if isinstance(port, str) else port
            if port_int in self.ports:
                self.ports[port_int].last_seen_episode = episode_num
                self.ports[port_int].confirmed_count += 1
            else:
                self.ports[port_int] = ConfirmedPort(
                    port=port_int,
                    first_seen_episode=episode_num,
                    last_seen_episode=episode_num,
                )
        
        # Record services
        for svc in discoveries.get("service", []):
            # Try to match to a port
            for port_obj in self.ports.values():
                if not port_obj.service and svc:
                    port_obj.service = svc
                    break
        
        # Record credentials
        if discoveries.get("credential"):
            # Check for duplicates
            cred_key = str(discoveries.get("credential"))
            if not any(
                c.username == cred_key or str(asdict(c)) == cred_key
                for c in self.credentials
            ):
                self.credentials.append(ConfirmedCredential(
                    username=cred_key,
                    password="",
                    first_seen_episode=episode_num,
                ))
        
        # Record vulns
        for cve in discoveries.get("cve", []):
            if cve not in self.vulns:
                self.vulns[cve] = ConfirmedVuln(
                    vuln_id=cve,
                    first_seen_episode=episode_num,
                )
            else:
                self.vulns[cve].confirmed_count += 1
        
        if discoveries.get("vulnerability") and not discoveries.get("cve"):
            vuln_key = f"vuln_ep{episode_num}"
            self.vulns[vuln_key] = ConfirmedVuln(
                vuln_id=vuln_key,
                vuln_type="generic",
                first_seen_episode=episode_num,
            )
        
        # Record shells
        if discoveries.get("shell"):
            self.shells.append({
                "episode": episode_num,
                "type": "root" if discoveries.get("root_shell") else "user",
                "timestamp": time.time(),
            })
        
        # Record attack chain if it's a good one
        if command_chain and highest_phase in phase_order:
            idx = phase_order.index(highest_phase)
            if idx >= 2:  # At least EXPLOITATION
                self.attack_chains.append(AttackChain(
                    episode=episode_num,
                    highest_phase=highest_phase,
                    steps=command_chain[:50],  # Cap at 50 steps
                    total_reward=total_reward,
                    discoveries_count=sum(
                        len(v) if isinstance(v, list) else 1
                        for v in discoveries.values()
                    ),
                ))
                # Sort by phase reached (descending), keep best 10
                self.attack_chains.sort(
                    key=lambda c: phase_order.index(c.highest_phase)
                    if c.highest_phase in phase_order else 0,
                    reverse=True,
                )
                self.attack_chains = self.attack_chains[:10]

    def record_failure(
        self,
        command_template: str,
        phase: str,
        episode_num: int,
        reason: str = "",
    ) -> None:
        """Record a consistently failing approach."""
        key = f"{command_template}@{phase}"
        if key in self.failed_approaches:
            self.failed_approaches[key].failure_count += 1
            self.failed_approaches[key].last_failure_episode = episode_num
        else:
            self.failed_approaches[key] = FailedApproach(
                command_template=command_template,
                phase=phase,
                failure_count=1,
                last_failure_episode=episode_num,
                reason=reason,
            )

    # ------------------------------------------------------------------
    # Querying prior knowledge
    # ------------------------------------------------------------------

    def get_prior_knowledge(self) -> Dict[str, Any]:
        """
        Get a summary of prior knowledge for injection into agent state.
        
        Returns dict suitable for merging into environment state or
        AttackContext for episode start.
        """
        return {
            "known_ports": sorted(self.ports.keys()),
            "known_services": {
                p.port: p.service
                for p in self.ports.values()
                if p.service
            },
            "known_credentials": len(self.credentials),
            "known_vulns": list(self.vulns.keys()),
            "shell_history": len(self.shells),
            "best_phase_ever": self.best_phase_reached,
            "total_prior_episodes": self.total_episodes,
            "best_chain_templates": (
                self.attack_chains[0].steps[:10]
                if self.attack_chains else []
            ),
            "failed_approaches": [
                f.command_template
                for f in self.failed_approaches.values()
                if f.failure_count >= 3
            ],
        }

    def get_known_ports_set(self) -> Set[int]:
        """Get set of confirmed open ports."""
        return set(self.ports.keys())

    def get_best_attack_chain(self) -> Optional[List[str]]:
        """Get the best recorded attack chain (highest phase, highest reward)."""
        if not self.attack_chains:
            return None
        return self.attack_chains[0].steps

    def should_skip_recon(self, threshold: int = 5) -> bool:
        """
        Check if we have enough prior knowledge to skip basic RECON.
        
        Returns True if we already know enough ports/services from
        prior episodes. Useful for curriculum acceleration.
        """
        return (
            len(self.ports) >= threshold
            and any(p.confirmed_count >= 2 for p in self.ports.values())
        )

    # ------------------------------------------------------------------
    # Context injection helpers
    # ------------------------------------------------------------------

    def inject_into_attack_context(self, attack_context: Any) -> None:
        """
        Inject prior knowledge into an AttackContext for episode start.
        
        This pre-populates the context so agents don't waste steps
        re-discovering known ports/services.
        """
        if not attack_context:
            return
        
        # Inject known ports
        for port_obj in self.ports.values():
            if port_obj.confirmed_count >= 2:  # Only inject well-confirmed facts
                attack_context.add_discovery("open_port", port_obj.port)
                if port_obj.service:
                    attack_context.add_service(port_obj.service, port_obj.port)
        
        # Inject known vulns
        for vuln in self.vulns.values():
            if vuln.confirmed_count >= 2:
                attack_context.add_discovery("vulnerability", vuln.vuln_id)
        
        # Set state flags based on prior knowledge
        if self.credentials:
            attack_context.set_state_flag("credentials_known")
        if any(s.get("type") == "root" for s in self.shells):
            attack_context.set_state_flag("shell_obtained")

    def get_mentor_context_block(self) -> str:
        """
        Generate a text block summarizing prior knowledge for mentor prompts.
        
        Injected into SmartMentor system/user prompt so the LLM knows
        what we've already confirmed in prior episodes.
        """
        if not self.ports and not self.credentials:
            return ""
        
        lines = ["## Prior Campaign Knowledge (from previous episodes)"]
        
        if self.ports:
            lines.append(f"Known open ports: {sorted(self.ports.keys())}")
            services = [
                f"  - Port {p.port}: {p.service} {p.version}".strip()
                for p in sorted(self.ports.values(), key=lambda x: x.port)
                if p.service
            ]
            if services:
                lines.extend(services)
        
        if self.credentials:
            lines.append(f"Known credentials: {len(self.credentials)} pairs confirmed")
        
        if self.vulns:
            lines.append(f"Known vulnerabilities: {list(self.vulns.keys())[:10]}")
        
        if self.attack_chains:
            best = self.attack_chains[0]
            lines.append(
                f"Best attack chain: reached {best.highest_phase} "
                f"in {len(best.steps)} steps (ep {best.episode})"
            )
        
        if self.failed_approaches:
            chronic = [
                f.command_template
                for f in self.failed_approaches.values()
                if f.failure_count >= 3
            ]
            if chronic:
                lines.append(f"Known failed approaches (avoid): {chronic[:5]}")
        
        return "\n".join(lines)
