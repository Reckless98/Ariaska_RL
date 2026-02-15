#!/usr/bin/env python3
"""
core/tracing/jsonl_logger.py — ARIASKA Multi-Brain Decision Logger v1.0

Structured JSONL logging for every decision step with full multi-brain telemetry.

Each line records:
  - episode#, step, agent, timestamp
  - macro_intent, macro_confidence (from DDQN)
  - PPO action, PPO confidence, PPO entropy
  - SAC action, SAC alpha
  - RND bonus (intrinsic novelty)
  - Codex persona, Codex response, Codex template
  - reward components (extrinsic, intrinsic, sil_bonus)
  - running total reward
  - winning_brain, brain_weights, fused_confidence
  - phase, discoveries, command

File is append-only and auto-rotated at 50MB.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger("ariaska.tracing.jsonl")


@dataclass
class DecisionLogEntry:
    """Schema for a single multi-brain decision log entry."""

    # ── Core identifiers ──
    episode: int = 0
    step: int = 0
    agent: str = ""
    timestamp: str = ""

    # ── Phase & command ──
    phase: str = ""
    command: str = ""
    template_name: str = ""
    source: str = "unknown"        # winning_brain or decision source

    # ── DDQN Macro ──
    macro_intent: Optional[str] = None
    macro_confidence: float = 0.0
    macro_q_max: float = 0.0

    # ── PPO Micro ──
    ppo_action: int = -1
    ppo_confidence: float = 0.0
    ppo_entropy: float = 0.0
    ppo_log_prob: float = 0.0
    ppo_value: float = 0.0

    # ── SAC Explorer ──
    sac_action: int = -1
    sac_confidence: float = 0.0
    sac_alpha: float = 0.0

    # ── RND Curiosity ──
    rnd_bonus: float = 0.0

    # ── SIL ──
    sil_triggered: bool = False
    sil_bonus: float = 0.0

    # ── EMA Critics ──
    ema_fast: float = 0.0
    ema_slow: float = 0.0

    # ── Codex Persona ──
    codex_role: Optional[str] = None
    codex_response: Optional[str] = None
    codex_template: Optional[str] = None
    codex_confidence: float = 0.0

    # ── Reward Components ──
    reward_extrinsic: float = 0.0
    reward_intrinsic: float = 0.0
    reward_total: float = 0.0
    reward_running_total: float = 0.0
    reward_discovery: float = 0.0
    reward_progress: float = 0.0
    reward_novelty: float = 0.0
    reward_redundancy: float = 0.0

    # ── Decision metadata ──
    confidence: float = 0.0       # Decision confidence from SmartCoach
    success: bool = False         # Whether command execution succeeded

    # ── Brain Fusion ──
    winning_brain: str = "unknown"
    cognition_brain: Optional[str] = None     # CognitionNode winning brain
    cognition_confidence: Optional[float] = None  # CognitionNode fused confidence
    fused_confidence: float = 0.0
    brain_weights: Optional[Dict[str, float]] = None

    # ── Discoveries ──
    new_discoveries: Optional[Dict[str, Any]] = None
    total_ports: int = 0
    total_services: int = 0
    total_shells: int = 0

    # ── Phase 10.0: KG + LLM counters ──
    kg_query_count: int = 0         # KnowledgeRetriever queries this step
    kg_hit_count: int = 0           # KR queries that returned >0 results
    kg_candidate_ids: Optional[List[str]] = None  # v2 candidate IDs used
    llm_calls_total: int = 0        # Total LLM calls this step (all models)
    llm_tokens_in: int = 0          # Input tokens consumed
    llm_tokens_out: int = 0         # Output tokens consumed
    llm_model_used: Optional[str] = None  # Primary model used
    llm_role_active: Optional[str] = None  # Cloud role that fired (if any)
    parser_stage: Optional[str] = None     # Parser broker stage that won
    venice_calls: int = 0           # Venice reasoning calls this step
    profile: Optional[str] = None   # Runtime profile (CLOUD/OFFLINE/DETERMINISTIC)

    def to_json(self) -> str:
        """Serialise to compact JSON string."""
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
        d = asdict(self)
        # Remove None values for compactness
        d = {k: v for k, v in d.items() if v is not None}
        return json.dumps(d, separators=(",", ":"), default=str)


class DecisionLogger:
    """
    Append-only JSONL logger for multi-brain decision telemetry.

    Usage:
        dl = DecisionLogger(log_dir="logs/decisions")
        dl.start_run(run_id="r76a")
        dl.log_decision(entry)  # Called every step
        dl.end_run()
    """

    MAX_FILE_SIZE_MB = 50  # Auto-rotate at this size

    def __init__(
        self,
        log_dir: str = "logs/decisions",
        enabled: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.enabled = enabled
        self._file = None
        self._file_path: Optional[Path] = None
        self._run_id: Optional[str] = None
        self._entries_written = 0
        self._running_reward: float = 0.0
        self._episode_reward: float = 0.0

        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def start_run(self, run_id: str = "default", config: Optional[Dict[str, Any]] = None) -> None:
        """Open a new JSONL log file for this training run."""
        if not self.enabled:
            return

        self._run_id = run_id
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._file_path = self.log_dir / f"{run_id}_{ts}.jsonl"

        try:
            self._file = open(self._file_path, "a", buffering=1)  # Line-buffered
            # Write header
            header = {
                "type": "run_start",
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "schema_version": "10.0",
                "config": config or {},
                "fields": [
                    "episode", "step", "agent", "phase", "command",
                    "macro_intent", "macro_confidence", "ppo_action",
                    "rnd_bonus", "codex_role", "codex_response",
                    "reward_extrinsic", "reward_intrinsic",
                    "reward_running_total", "winning_brain",
                ],
            }
            self._file.write(json.dumps(header) + "\n")
            self._entries_written = 0
            logger.info(f"DecisionLogger started: {self._file_path}")
        except Exception as e:
            logger.warning(f"Failed to open decision log: {e}")
            self._file = None

    def start_episode(self, episode: int) -> None:
        """Reset per-episode state."""
        self._episode_reward = 0.0
        if self._file:
            entry = {
                "type": "episode_start",
                "episode": episode,
                "timestamp": datetime.now().isoformat(),
            }
            self._file.write(json.dumps(entry) + "\n")

    def log_decision(self, entry: DecisionLogEntry) -> None:
        """Write a single decision entry to the JSONL log."""
        if not self.enabled or self._file is None:
            return

        # Update running totals
        self._episode_reward += entry.reward_total
        self._running_reward += entry.reward_total
        entry.reward_running_total = round(self._running_reward, 2)

        try:
            self._file.write(entry.to_json() + "\n")
            self._entries_written += 1

            # Auto-rotate check
            if self._entries_written % 1000 == 0:
                self._check_rotate()

        except Exception as e:
            logger.debug(f"Failed to write decision log: {e}")

    def end_episode(self, episode: int = -1, total_reward: float = 0.0, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Write episode summary."""
        if metrics is not None:
            total_reward = metrics.get("total_reward", total_reward)
            episode = metrics.get("episode", episode)
        if self._file:
            entry = {
                "type": "episode_end",
                "episode": episode,
                "total_reward": round(total_reward, 2),
                "timestamp": datetime.now().isoformat(),
            }
            if metrics:
                entry["metrics"] = {k: v for k, v in metrics.items() if k not in ("total_reward", "episode")}
            self._file.write(json.dumps(entry, default=str) + "\n")

    def end_run(self, final_metrics: Optional[Dict[str, Any]] = None) -> None:
        """Close the log file."""
        if self._file:
            try:
                summary = {
                    "type": "run_end",
                    "run_id": self._run_id,
                    "entries_written": self._entries_written,
                    "total_reward": round(self._running_reward, 2),
                    "timestamp": datetime.now().isoformat(),
                }
                self._file.write(json.dumps(summary) + "\n")
                self._file.close()
                logger.info(
                    f"DecisionLogger ended: {self._entries_written} entries, "
                    f"total reward {self._running_reward:.1f}"
                )
            except Exception:
                pass
            finally:
                self._file = None

    def _check_rotate(self) -> None:
        """Rotate log file if it exceeds MAX_FILE_SIZE_MB."""
        if self._file_path and self._file_path.exists():
            size_mb = self._file_path.stat().st_size / (1024 * 1024)
            if size_mb > self.MAX_FILE_SIZE_MB:
                logger.info(f"Rotating decision log ({size_mb:.1f}MB)")
                self.end_run()
                self.start_run(self._run_id or "rotated")

    def __del__(self):
        """Cleanup on garbage collection."""
        if self._file:
            try:
                self._file.close()
            except Exception:
                pass


def create_decision_logger(
    log_dir: str = "logs/decisions",
    enabled: bool = True,
) -> DecisionLogger:
    """Factory function for DecisionLogger."""
    return DecisionLogger(log_dir=log_dir, enabled=enabled)
