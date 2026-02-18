#!/usr/bin/env python3
"""
core/training/checkpoint_manager.py — ARIASKA Checkpoint Manager v1.0

Safe, atomic checkpoint saves with periodic auto-save and target health monitoring.

Features:
- Atomic writes (temp file + os.rename) to prevent corruption
- Periodic auto-checkpoint every N episodes
- Target health ping before each episode (for live mode)
- Crash recovery via --resume flag
"""

import os
import json
import time
import shutil
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger("ariaska.checkpoint")


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management."""
    checkpoint_dir: str = "models/enhanced"
    checkpoint_name: str = "ppo_checkpoint.pt"
    auto_save_interval: int = 5         # Save every N episodes
    keep_last_n: int = 3                # Keep last N checkpoint versions
    target_health_enabled: bool = False # Ping target before episodes
    target_ip: str = ""
    health_timeout: float = 5.0         # Seconds to wait for ping


class CheckpointManager:
    """
    Manages safe checkpoint saves, auto-save scheduling, and target health monitoring.
    
    Phase 6.3: Extended with full training state serialization
    (episode counter, mentor state, campaign memory, etc.).
    """

    def __init__(self, config: Optional[CheckpointConfig] = None):
        self.config = config or CheckpointConfig()
        self._checkpoint_dir = Path(self.config.checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._last_save_episode: int = -1
        self._save_count: int = 0

    # ------------------------------------------------------------------
    # Atomic save
    # ------------------------------------------------------------------

    def save_atomic(
        self,
        state_dict: Any,
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a PyTorch state dict atomically (temp file + rename).

        Args:
            state_dict: PyTorch model state dict or any picklable object
            filename: Override filename (default: config.checkpoint_name)
            metadata: Optional metadata dict saved alongside

        Returns:
            Absolute path to saved checkpoint
        """
        import torch

        fname = filename or self.config.checkpoint_name
        target_path = self._checkpoint_dir / fname

        # Write to temp file first
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._checkpoint_dir),
            prefix=".ckpt_tmp_",
            suffix=".pt",
        )
        try:
            os.close(fd)
            save_payload = {
                "state_dict": state_dict,
                "timestamp": time.time(),
                "save_count": self._save_count,
            }
            if metadata:
                save_payload["metadata"] = metadata
            torch.save(save_payload, tmp_path)

            # Atomic rename
            os.replace(tmp_path, str(target_path))
            self._save_count += 1

            logger.info(f"[CKPT] Atomic save → {target_path} (#{self._save_count})")
            return str(target_path)

        except Exception as e:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            logger.error(f"[CKPT] Save failed: {e}")
            raise

    def save_versioned(
        self,
        state_dict: Any,
        episode: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a versioned checkpoint (e.g. ppo_checkpoint_ep050.pt).
        Cleans up old versions beyond keep_last_n.
        """
        base = Path(self.config.checkpoint_name).stem
        suffix = Path(self.config.checkpoint_name).suffix or ".pt"
        versioned_name = f"{base}_ep{episode:04d}{suffix}"

        path = self.save_atomic(state_dict, filename=versioned_name, metadata=metadata)

        # Also save as "latest"
        self.save_atomic(state_dict, metadata=metadata)

        # Cleanup old versions
        self._cleanup_old_versions(base, suffix)

        return path

    def _cleanup_old_versions(self, base: str, suffix: str) -> None:
        """Remove old versioned checkpoints beyond keep_last_n."""
        import re
        pattern = re.compile(rf"^{re.escape(base)}_ep(\d+){re.escape(suffix)}$")
        versions = []
        for f in self._checkpoint_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                versions.append((int(m.group(1)), f))

        versions.sort(key=lambda x: x[0], reverse=True)
        for _, old_path in versions[self.config.keep_last_n:]:
            try:
                old_path.unlink()
                logger.debug(f"[CKPT] Removed old checkpoint: {old_path.name}")
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Auto-save check
    # ------------------------------------------------------------------

    def should_auto_save(self, episode: int) -> bool:
        """Check if auto-save should trigger at this episode."""
        if self.config.auto_save_interval <= 0:
            return False
        return (episode > 0) and (episode % self.config.auto_save_interval == 0)

    def auto_save(
        self,
        state_dict: Any,
        episode: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Auto-save if interval is reached. Returns path if saved, None otherwise.
        """
        if self.should_auto_save(episode):
            self._last_save_episode = episode
            return self.save_versioned(state_dict, episode, metadata=metadata)
        return None

    # ------------------------------------------------------------------
    # Load / Resume
    # ------------------------------------------------------------------

    def load(self, path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint.

        Args:
            path: Explicit path, or None to load from default location

        Returns:
            Dict with 'state_dict', 'metadata', 'timestamp', or None if not found
        """
        import torch

        if path:
            ckpt_path = Path(path)
        else:
            ckpt_path = self._checkpoint_dir / self.config.checkpoint_name

        if not ckpt_path.exists():
            logger.warning(f"[CKPT] No checkpoint at {ckpt_path}")
            return None

        try:
            data = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            logger.info(
                f"[CKPT] Loaded checkpoint from {ckpt_path} "
                f"(save #{data.get('save_count', '?')})"
            )
            return data
        except Exception as e:
            logger.error(f"[CKPT] Failed to load {ckpt_path}: {e}")
            return None

    # ------------------------------------------------------------------
    # Target health monitoring
    # ------------------------------------------------------------------

    def check_target_health(self, target_ip: Optional[str] = None) -> bool:
        """
        Ping the target to verify it's alive.

        Args:
            target_ip: Override target IP

        Returns:
            True if target responds to ping
        """
        ip = target_ip or self.config.target_ip
        if not ip or not self.config.target_health_enabled:
            return True  # Skip if not configured

        try:
            result = subprocess.run(
                ["ping", "-c", "1", "-W", str(int(self.config.health_timeout)), ip],
                capture_output=True,
                timeout=self.config.health_timeout + 2,
            )
            alive = result.returncode == 0
            if not alive:
                logger.warning(f"[HEALTH] Target {ip} not responding to ping!")
            return alive
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.warning(f"[HEALTH] Ping failed for {ip}: {e}")
            return False

    def wait_for_target(
        self,
        target_ip: Optional[str] = None,
        max_retries: int = 5,
        retry_interval: float = 10.0,
    ) -> bool:
        """
        Wait for target to become available, with retries.

        Returns True if target is reachable, False if all retries exhausted.
        """
        ip = target_ip or self.config.target_ip
        for attempt in range(max_retries):
            if self.check_target_health(ip):
                return True
            if attempt < max_retries - 1:
                logger.warning(
                    f"[HEALTH] Retry {attempt + 1}/{max_retries} for {ip} "
                    f"in {retry_interval}s..."
                )
                time.sleep(retry_interval)
        return False

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return {
            "save_count": self._save_count,
            "last_save_episode": self._last_save_episode,
            "checkpoint_dir": str(self._checkpoint_dir),
            "auto_save_interval": self.config.auto_save_interval,
        }

    # ------------------------------------------------------------------
    # Phase 6.3: Full training state save/load
    # ------------------------------------------------------------------

    def save_full_state(
        self,
        episode: int,
        orchestrator: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save the FULL training state (not just PPO weights).
        
        Persists: episode counter, PPO weights for all coaches,
        MentorController state, SkillLibrary, campaign memory,
        and training config.
        
        Args:
            episode: Current episode number
            orchestrator: SmartOrchestrator instance
            metadata: Optional metadata
            
        Returns:
            Path to saved checkpoint
        """
        import torch

        full_state: Dict[str, Any] = {
            "version": "6.3",
            "episode": episode,
            "timestamp": time.time(),
        }

        # 1. PPO weights for all coaches
        ppo_states = {}
        for coach_name, coach in orchestrator.coaches.items():
            if hasattr(coach, 'ppo_agent') and coach.ppo_agent is not None:
                ppo_states[coach_name] = coach.ppo_agent.state_dict()
        full_state["ppo_states"] = ppo_states

        # 2. MentorController state
        if hasattr(orchestrator, 'mentor_controller') and orchestrator.mentor_controller:
            mc = orchestrator.mentor_controller
            full_state["mentor_controller"] = {
                "total_calls": getattr(mc, '_total_calls', 0),
                "episode_calls": getattr(mc, '_episode_calls', 0),
                "consecutive_no_exfil": getattr(mc, '_consecutive_no_exfil', 0),
                "episodes_seen": getattr(mc, '_episodes_seen', 0),
            }

        # 3. Skill library snapshot path (it saves itself to JSON)
        if orchestrator.skill_library:
            try:
                orchestrator.skill_library._save_library()
                full_state["skill_library_path"] = str(
                    getattr(orchestrator.skill_library, '_path', 'data/skill_library.json')
                )
            except Exception:
                pass

        # 4. Campaign memory snapshot path
        if hasattr(orchestrator, 'campaign_memory') and orchestrator.campaign_memory:
            try:
                orchestrator.campaign_memory.save()
                full_state["campaign_memory_path"] = str(
                    orchestrator.campaign_memory._path
                )
            except Exception:
                pass

        # 5. Training config
        if orchestrator.config:
            full_state["config"] = {
                "max_steps_per_episode": orchestrator.config.max_steps_per_episode,
                "mentor_budget_pct": getattr(orchestrator.config, 'mentor_budget_pct', 0.3),
                "model": orchestrator.config.model,
            }

        if metadata:
            full_state["metadata"] = metadata

        return self.save_versioned(full_state, episode, metadata=metadata)

    def load_full_state(
        self,
        orchestrator: Any,
        path: Optional[str] = None,
    ) -> Optional[int]:
        """
        Restore full training state into an orchestrator.
        
        Args:
            orchestrator: SmartOrchestrator instance to restore into
            path: Explicit checkpoint path, or None for latest
            
        Returns:
            Episode number to resume from, or None if no checkpoint
        """
        data = self.load(path)
        if not data:
            return None

        state = data.get("state_dict", data)  # Handle both formats
        episode = state.get("episode", 0)

        # 1. Restore PPO weights
        ppo_states = state.get("ppo_states", {})
        for coach_name, sd in ppo_states.items():
            if coach_name in orchestrator.coaches:
                coach = orchestrator.coaches[coach_name]
                if hasattr(coach, 'ppo_agent') and coach.ppo_agent is not None:
                    try:
                        coach.ppo_agent.load_state_dict(sd)
                        logger.info(f"[CKPT] Restored PPO for {coach_name}")
                    except Exception as e:
                        logger.warning(f"[CKPT] Failed to restore PPO for {coach_name}: {e}")

        # 2. Restore MentorController state
        mc_state = state.get("mentor_controller", {})
        if mc_state and hasattr(orchestrator, 'mentor_controller') and orchestrator.mentor_controller:
            mc = orchestrator.mentor_controller
            for key, val in mc_state.items():
                if hasattr(mc, f"_{key}"):
                    setattr(mc, f"_{key}", val)

        # 3. Skill library (re-load from its own file — already done in __init__)
        # Just log that it's available
        if state.get("skill_library_path") and orchestrator.skill_library:
            logger.info(f"[CKPT] SkillLibrary available from {state['skill_library_path']}")

        # 4. Campaign memory (re-load from its own file — already done in __init__)
        if state.get("campaign_memory_path") and hasattr(orchestrator, 'campaign_memory'):
            logger.info(f"[CKPT] CampaignMemory available from {state['campaign_memory_path']}")

        logger.info(f"[CKPT] Full state restored — resume from episode {episode}")
        return episode
