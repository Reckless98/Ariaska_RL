"""Unified checkpoint format for Ariaska RL — v2.

Every checkpoint saved anywhere in the system (GPU distillation, local
SmartOrchestrator, enhanced per-agent) goes through this module.  On
load, legacy formats (v0/v1 PPO-only, agent-brain, bare DDQN) are
transparently migrated so that old files keep working.

Directory layout (post-unification)::

    models/
    ├── unified/                     # ← primary checkpoint dir
    │   ├── ariaska_<run_id>_ep<N>.pt
    │   └── ...
    ├── distilled/                   # legacy GPU distilled (auto-migrated)
    │   └── h200_<run_id>_ep<N>.pt
    └── enhanced/                    # legacy per-agent (auto-migrated)
        ├── ppo_live_checkpoint.pt/
        └── episode_*/
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.checkpoints")

# ── Constants ─────────────────────────────────────────────────────────────
FORMAT_VERSION = 2
UNIFIED_DIR = Path("models/unified")

# Legacy path patterns
_DISTILLED_RE = re.compile(r"h200_(.+?)_ep(\d+)\.pt$")
_GRPO_RE = re.compile(r"grpo_(.+?)_(ep\d+|final)\.pt$")


# ── Unified Checkpoint ────────────────────────────────────────────────────

@dataclass
class UnifiedCheckpoint:
    """Single file capturing ALL algorithm states from a training run.

    Attributes:
        format_version: Schema version (always ``FORMAT_VERSION``).
        timestamp: ISO-8601 save time.
        run_id: Training run identifier.
        episode: Episode number at save time.
        source: Origin tag — ``gpu_distill``, ``local_train``, ``enhanced``,
                ``migrated``.
        ppo_state: Full PPO Agent state dict (``PPOAgent.save()`` format).
        ddqn_states: Per-agent DDQN macro states keyed by agent name.
        sac_state: SAC agent state dict (if trained).
        agent_states: Per-agent brain states (RedAgent ``policy_network_state``
                      / ``value_network_state`` format).
        metadata: Free-form training metadata (rewards, scenarios, etc.).
    """

    format_version: int = FORMAT_VERSION
    timestamp: str = ""
    run_id: str = ""
    episode: int = 0
    source: str = "unknown"

    # Algorithm states — any can be None / empty
    ppo_state: Optional[Dict[str, Any]] = None
    ddqn_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    sac_state: Optional[Dict[str, Any]] = None
    agent_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Training metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Save ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> str:
        """Persist to disk as a torch file.

        Returns:
            Absolute path of the saved file.
        """
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        payload: Dict[str, Any] = {
            "__unified_checkpoint__": True,
            "format_version": self.format_version,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "episode": self.episode,
            "source": self.source,
            "metadata": self.metadata,
        }

        if self.ppo_state is not None:
            payload["ppo_state"] = self.ppo_state
        if self.ddqn_states:
            payload["ddqn_states"] = self.ddqn_states
        if self.sac_state is not None:
            payload["sac_state"] = self.sac_state
        if self.agent_states:
            payload["agent_states"] = self.agent_states

        torch.save(payload, str(path))
        logger.info(
            "Unified checkpoint saved: %s  (ep=%d, source=%s, ppo=%s, ddqn=%d, sac=%s, agents=%d)",
            path.name,
            self.episode,
            self.source,
            "yes" if self.ppo_state else "no",
            len(self.ddqn_states),
            "yes" if self.sac_state else "no",
            len(self.agent_states),
        )
        return str(path.resolve())

    # ── Load ──────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str | Path) -> "UnifiedCheckpoint":
        """Load from disk — handles unified v2 and ALL legacy formats.

        Raises:
            FileNotFoundError: If *path* doesn't exist.
            ValueError: If the file is unrecognizable.
        """
        import torch

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        data = torch.load(str(path), map_location="cpu", weights_only=False)

        # ── Unified format ────────────────────────────────────────────
        if isinstance(data, dict) and data.get("__unified_checkpoint__"):
            return cls(
                format_version=data.get("format_version", FORMAT_VERSION),
                timestamp=data.get("timestamp", ""),
                run_id=data.get("run_id", ""),
                episode=data.get("episode", 0),
                source=data.get("source", "unknown"),
                ppo_state=data.get("ppo_state"),
                ddqn_states=data.get("ddqn_states", {}),
                sac_state=data.get("sac_state"),
                agent_states=data.get("agent_states", {}),
                metadata=data.get("metadata", {}),
            )

        # ── Legacy PPO format (distilled / enhanced) ──────────────────
        if isinstance(data, dict) and "network_state_dict" in data:
            run_id, episode = _extract_run_ep(path)
            return cls(
                format_version=1,
                timestamp="",
                run_id=run_id,
                episode=episode,
                source="legacy_ppo",
                ppo_state=data,
                metadata={
                    "total_steps": data.get("total_steps", 0),
                    "updates_done": data.get("updates_done", 0),
                    "migrated_from": str(path),
                },
            )

        # ── Legacy DDQN format ────────────────────────────────────────
        if isinstance(data, dict) and "online_net" in data and "target_net" in data:
            agent_name = _guess_agent_name(path)
            return cls(
                format_version=1,
                source="legacy_ddqn",
                ddqn_states={agent_name: data},
                metadata={"migrated_from": str(path)},
            )

        # ── Legacy SAC format ─────────────────────────────────────────
        if isinstance(data, dict) and "actor" in data and "critic" in data:
            return cls(
                format_version=1,
                source="legacy_sac",
                sac_state=data,
                metadata={"migrated_from": str(path)},
            )

        # ── Legacy Agent-brain format (episode_*/red_agent) ───────────
        if isinstance(data, dict) and "policy_network_state" in data:
            agent_name = _guess_agent_name(path)
            return cls(
                format_version=1,
                source="legacy_agent_brain",
                agent_states={agent_name: data},
                metadata={"migrated_from": str(path)},
            )

        raise ValueError(
            f"Unrecognized checkpoint format at {path}: "
            f"keys={list(data.keys())[:10] if isinstance(data, dict) else type(data)}"
        )

    # ── Apply to live agents ──────────────────────────────────────────

    def apply_ppo(self, ppo_agent: Any) -> bool:
        """Load PPO state into a live ``PPOAgent`` instance.

        Uses ``PPOAgent.load_from_state_dict()`` for direct in-memory
        loading without temp files.

        Returns True on success, False if no PPO state is available or
        loading failed.
        """
        if self.ppo_state is None:
            return False
        try:
            ppo_agent.load_from_state_dict(self.ppo_state)
            return True
        except Exception as e:
            logger.warning("Failed to apply PPO state: %s", e)
            return False

    def apply_ddqn(self, ddqn_macro: Any, agent_name: str) -> bool:
        """Load DDQN state for *agent_name* into a live ``DDQNMacro``."""
        state = self.ddqn_states.get(agent_name)
        if state is None:
            return False
        try:
            ddqn_macro.load_state_dict(state)
            return True
        except Exception as e:
            logger.warning("Failed to apply DDQN state for %s: %s", agent_name, e)
            return False

    def apply_sac(self, sac_agent: Any) -> bool:
        """Load SAC state into a live ``SACAgent``.

        SAC doesn't have ``load_from_state_dict`` yet, so we use a temp
        file round-trip via its native ``load()``.
        """
        if self.sac_state is None:
            return False
        import tempfile
        import torch
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                tmp = f.name
                torch.save(self.sac_state, tmp)
            sac_agent.load(tmp)
            return True
        except Exception as e:
            logger.warning("Failed to apply SAC state: %s", e)
            return False
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except Exception:
                    pass

    def apply_agent_brains(self, coaches: Dict[str, Any]) -> int:
        """Restore per-agent brain network weights from ``agent_states``.

        Returns:
            Number of agents whose brain weights were restored.
        """
        if not self.agent_states:
            return 0

        restored = 0
        for name, brain_state in self.agent_states.items():
            coach = coaches.get(name)
            if coach is None:
                continue
            agent = getattr(coach, "agent", None)
            if agent is None:
                continue

            brain = getattr(agent, "brain", None)
            if brain is not None:
                if "policy_network_state" in brain_state and hasattr(brain, "policy_network"):
                    try:
                        brain.policy_network.load_state_dict(brain_state["policy_network_state"])
                        restored += 1
                    except Exception as e:
                        logger.warning("Failed to restore %s brain policy: %s", name, e)
                if "value_network_state" in brain_state and hasattr(brain, "value_network"):
                    try:
                        brain.value_network.load_state_dict(brain_state["value_network_state"])
                    except Exception as e:
                        logger.warning("Failed to restore %s brain value: %s", name, e)

        return restored

    # ── Convenience builders ──────────────────────────────────────────

    @classmethod
    def from_ppo_agent(
        cls,
        ppo_agent: Any,
        run_id: str = "",
        episode: int = 0,
        source: str = "local_train",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "UnifiedCheckpoint":
        """Capture current PPO state without saving to disk.

        Builds the state dict directly from PPO internals instead of
        round-tripping through ``save()`` + ``torch.load()``.
        """
        state = _capture_ppo_state(ppo_agent)

        return cls(
            run_id=run_id,
            episode=episode,
            source=source,
            ppo_state=state,
            metadata=metadata or {},
        )

    @classmethod
    def from_coaches(
        cls,
        coaches: Dict[str, Any],
        run_id: str = "",
        episode: int = 0,
        source: str = "local_train",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "UnifiedCheckpoint":
        """Capture ALL algorithm states from all SmartCoach instances.

        This is the canonical way to save a complete snapshot of the
        local orchestrator.  It grabs PPO, DDQN, SAC, and agent brain
        states so that reloading reproduces the exact same state.
        """
        ppo_state: Optional[Dict[str, Any]] = None
        ddqn_states: Dict[str, Dict[str, Any]] = {}
        sac_state: Optional[Dict[str, Any]] = None
        agent_states: Dict[str, Dict[str, Any]] = {}

        for name, coach in coaches.items():
            # PPO — all coaches share the same PPO, capture once
            if ppo_state is None and hasattr(coach, "ppo_agent") and coach.ppo_agent is not None:
                ppo_state = _capture_ppo_state(coach.ppo_agent)

            # DDQN — per-agent
            if hasattr(coach, "ddqn_macro") and coach.ddqn_macro is not None:
                try:
                    ddqn_states[name] = coach.ddqn_macro.state_dict()
                except Exception:
                    pass

            # SAC — capture once
            if sac_state is None and hasattr(coach, "sac_agent") and coach.sac_agent is not None:
                sac_state = _capture_sac_state(coach.sac_agent)

            # Agent brain state — capture per-agent network weights
            agent = getattr(coach, "agent", None)
            if agent is not None:
                brain_state = _capture_agent_brain(agent)
                if brain_state:
                    agent_states[name] = brain_state

        return cls(
            run_id=run_id,
            episode=episode,
            source=source,
            ppo_state=ppo_state,
            ddqn_states=ddqn_states,
            sac_state=sac_state,
            agent_states=agent_states,
            metadata=metadata or {},
        )

    # ── Scanning / discovery ──────────────────────────────────────────

    @staticmethod
    def find_best(
        directories: Optional[List[str | Path]] = None,
        run_id: Optional[str] = None,
    ) -> Optional[Path]:
        """Find the best checkpoint across unified + legacy directories.

        Priority:
          1. Unified dir (``models/unified/``) — highest episode
          2. Distilled dir (``models/distilled/``) — highest episode of latest run
          3. Enhanced dir (``models/enhanced/ppo_live_checkpoint.pt/``)

        Returns:
            Path to the best checkpoint file, or None.
        """
        if directories is None:
            directories = [
                UNIFIED_DIR,
                Path("models/distilled"),
                Path("models/enhanced"),
            ]

        best_path: Optional[Path] = None
        best_priority: int = -1

        for d in directories:
            d = Path(d)
            if not d.is_dir():
                continue

            # Unified format files
            for f in d.iterdir():
                if f.suffix == ".pt" and f.name.startswith("ariaska_"):
                    m = re.search(r"_ep(\d+)\.pt$", f.name)
                    ep = int(m.group(1)) if m else 0
                    pri = 10000 + ep  # unified always wins
                    if run_id and run_id not in f.name:
                        continue
                    if pri > best_priority:
                        best_priority = pri
                        best_path = f

            # Legacy distilled
            for f in d.iterdir():
                if f.suffix == ".pt" and f.name.startswith("h200_"):
                    m = _DISTILLED_RE.match(f.name)
                    if m:
                        rid = m.group(1)
                        ep = int(m.group(2))
                        if run_id and run_id != rid:
                            continue
                        pri = 5000 + ep
                        if pri > best_priority:
                            best_priority = pri
                            best_path = f

            # Legacy enhanced (ppo_live > ppo_sim)
            for sub in d.iterdir():
                if not sub.is_dir():
                    continue
                for pt in sub.iterdir():
                    if pt.suffix == ".pt" and pt.name.startswith("ppo_"):
                        if "live" in sub.name:
                            pri = 200
                        elif "sim" in sub.name:
                            pri = 100
                        else:
                            pri = 50
                        if pri > best_priority:
                            best_priority = pri
                            best_path = pt

        return best_path

    # ── Summary ───────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable one-line summary."""
        parts = [f"v{self.format_version}"]
        parts.append(f"run={self.run_id}" if self.run_id else "no-run")
        parts.append(f"ep={self.episode}")
        parts.append(f"src={self.source}")

        algos = []
        if self.ppo_state:
            steps = self.ppo_state.get("total_steps", 0)
            updates = self.ppo_state.get("updates_done", 0)
            algos.append(f"PPO({steps:,}steps/{updates}upd)")
        if self.ddqn_states:
            for name, st in self.ddqn_states.items():
                s = st.get("total_steps", 0)
                algos.append(f"DDQN-{name}({s:,})")
        if self.sac_state:
            s = self.sac_state.get("step_count", 0)
            algos.append(f"SAC({s:,})")
        if self.agent_states:
            algos.append(f"agents={list(self.agent_states.keys())}")

        parts.append(" + ".join(algos) if algos else "empty")
        return " | ".join(parts)


# ── Migration helpers ─────────────────────────────────────────────────────

def migrate_directory(
    src_dir: str | Path,
    dst_dir: str | Path = UNIFIED_DIR,
    *,
    remove_originals: bool = False,
) -> int:
    """Migrate all legacy checkpoints in *src_dir* to unified format.

    Scans for ``.pt`` files and ``red_agent``/``ddqn_*`` files, loads
    each via ``UnifiedCheckpoint.load()``, and re-saves into *dst_dir*.

    Returns:
        Number of files migrated.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    migrated = 0

    for root, _dirs, files in os.walk(src_dir):
        root_p = Path(root)
        for fname in files:
            fpath = root_p / fname
            # Skip non-torch files
            if fpath.suffix not in {".pt", ""} or fname.startswith("."):
                if fname not in {"red_agent"}:
                    continue

            # Skip already-unified files
            if fname.startswith("ariaska_"):
                continue

            try:
                ckpt = UnifiedCheckpoint.load(fpath)
            except Exception as e:
                logger.debug("Skipping %s: %s", fpath, e)
                continue

            # Build unified filename
            run = ckpt.run_id or "migrated"
            ep = ckpt.episode
            tag = f"ariaska_{run}_ep{ep:04d}"
            # Disambiguate
            dst_path = dst_dir / f"{tag}.pt"
            suffix_n = 0
            while dst_path.exists():
                suffix_n += 1
                dst_path = dst_dir / f"{tag}_{suffix_n}.pt"

            ckpt.source = "migrated"
            ckpt.format_version = FORMAT_VERSION
            ckpt.save(dst_path)
            migrated += 1
            logger.info("Migrated %s → %s", fpath, dst_path.name)

            if remove_originals:
                fpath.unlink()

    return migrated


# ── Private helpers ───────────────────────────────────────────────────────


def _capture_ppo_state(ppo_agent: Any) -> Dict[str, Any]:
    """Build PPO state dict directly from agent internals.

    Mirrors ``PPOAgent.save()`` but returns the dict instead of
    writing to disk — avoids temp-file round-trips entirely.
    """
    import torch

    state: Dict[str, Any] = {
        "network_state_dict": ppo_agent.network.state_dict(),
        "optimizer_state_dict": ppo_agent.optimizer.state_dict(),
        "scheduler_state_dict": ppo_agent.lr_scheduler.state_dict(),
        "total_steps": ppo_agent.total_steps,
        "updates_done": ppo_agent.updates_done,
        "entropy_coef": ppo_agent.entropy_coef,
        "config": ppo_agent.config,
        "training_metrics": ppo_agent.training_metrics,
    }

    # Phase 6: reward normalisation
    state["reward_norm"] = {
        "mean": ppo_agent._reward_mean,
        "var": ppo_agent._reward_var,
        "count": ppo_agent._reward_count,
        "return_mean": ppo_agent._return_mean,
        "return_var": ppo_agent._return_var,
        "return_count": ppo_agent._return_count,
    }

    # R58 Layer 2c: adaptive entropy
    state["adaptive_entropy"] = {
        "multiplier": ppo_agent._entropy_adaptive_multiplier,
        "consecutive_closeouts": ppo_agent._consecutive_closeouts,
        "consecutive_failures": ppo_agent._consecutive_failures,
    }

    # R68: phase gates
    state["has_phase_gates"] = getattr(ppo_agent.network, "has_phase_gates", False)

    # R70: SIL buffer
    state["sil_baseline"] = ppo_agent.sil_buffer._return_baseline
    state["sil_count"] = ppo_agent.sil_buffer._return_count

    # R71: symlog + cosine entropy
    state["use_symlog"] = ppo_agent.config.use_symlog
    state["use_cosine_entropy"] = ppo_agent.config.use_cosine_entropy

    # R75: EMA network
    state["ema_network_state"] = (
        ppo_agent.ema_network.state_dict() if ppo_agent.ema_network else None
    )

    # R80: adaptive clip + entropy rebound
    state["clip_epsilon_current"] = ppo_agent.config.clip_epsilon
    state["clip_fraction_history"] = ppo_agent._clip_fraction_history
    state["entropy_below_count"] = ppo_agent._entropy_below_count

    return state


def _capture_sac_state(sac_agent: Any) -> Optional[Dict[str, Any]]:
    """Capture SAC state via temp file (SAC has no direct state_dict API)."""
    import tempfile
    import torch

    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp = f.name
        sac_agent.save(tmp)
        state = torch.load(tmp, map_location="cpu", weights_only=False)
        return state
    except Exception as e:
        logger.warning("Failed to capture SAC state: %s", e)
        return None
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


def _capture_agent_brain(agent: Any) -> Optional[Dict[str, Any]]:
    """Capture agent brain network weights if available.

    Agents like RedAgent store ``policy_network_state`` and
    ``value_network_state`` for their brain networks.
    """
    brain_state: Dict[str, Any] = {}

    # RedAgent brain pattern
    brain = getattr(agent, "brain", None)
    if brain is not None:
        if hasattr(brain, "policy_network") and brain.policy_network is not None:
            try:
                brain_state["policy_network_state"] = brain.policy_network.state_dict()
            except Exception:
                pass
        if hasattr(brain, "value_network") and brain.value_network is not None:
            try:
                brain_state["value_network_state"] = brain.value_network.state_dict()
            except Exception:
                pass

    # Direct agent pattern (policy_net / value_net on agent itself)
    if not brain_state:
        for attr in ("policy_net", "policy_network", "value_net", "value_network"):
            net = getattr(agent, attr, None)
            if net is not None and hasattr(net, "state_dict"):
                try:
                    brain_state[f"{attr}_state"] = net.state_dict()
                except Exception:
                    pass

    return brain_state if brain_state else None


def _extract_run_ep(path: Path) -> tuple[str, int]:
    """Extract run_id and episode from filename patterns."""
    name = path.name
    m = _DISTILLED_RE.match(name)
    if m:
        return m.group(1), int(m.group(2))
    m = _GRPO_RE.match(name)
    if m:
        ep_str = m.group(2)
        ep = int(ep_str.replace("ep", "")) if ep_str != "final" else 9999
        return m.group(1), ep
    # Enhanced per-agent: try parent dir
    parent = path.parent.name
    ep_m = re.search(r"episode_(\d+)", parent)
    if ep_m:
        return parent, int(ep_m.group(1))
    return path.stem, 0


def _guess_agent_name(path: Path) -> str:
    """Guess agent name from file path."""
    name = path.stem
    for agent in ("RedAgent", "BlueAgent", "ScoutAgent", "ShadowAgent", "OrionAgent"):
        if agent.lower() in name.lower() or agent.lower() in str(path).lower():
            return agent
    if name in {"red_agent", "red"}:
        return "RedAgent"
    return "UnknownAgent"
