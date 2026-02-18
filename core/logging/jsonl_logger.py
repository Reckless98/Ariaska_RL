"""
JSONL Logger + Live HUD for Ariaska training runs.

R66: Provides:
  - Per-step JSONL records (phase, source, macro, coherence, reward, intrinsic, etc.)
  - Per-episode summary JSONL records
  - Compact live HUD line per step printed to terminal
  - Episode summary block printed to terminal

Usage:
    from core.logging.jsonl_logger import RunLogger
    rl = RunLogger(run_tag="r66_ms2", log_dir="logs")
    rl.log_step(ep=1, step=5, phase="EXPLOITATION", ...)
    rl.log_episode_summary(ep=1, ...)
    rl.print_hud_line(...)
    rl.print_episode_summary(...)
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.logging.jsonl")


@dataclass
class StepRecord:
    """Single step log record for JSONL."""
    ep: int
    step: int
    phase: str
    source: str
    macro: str = ""
    macro_conf: float = 0.0
    coherence: float = 0.0
    reward_delta: float = 0.0
    intrinsic_reward: float = 0.0
    anti_repeat_fired: bool = False
    codex_fired: bool = False
    codex_role: str = ""           # tactical / strategic / ""
    template_name: str = ""
    command_prefix: str = ""
    agent: str = ""
    record_type: str = "step"


@dataclass
class EpisodeSummary:
    """Episode summary record for JSONL."""
    ep: int
    total_reward: float
    steps: int
    highest_phase: str
    closeout: bool = False
    sources: Dict[str, int] = field(default_factory=dict)
    discoveries: int = 0
    unique_commands: int = 0
    codex_templates: List[str] = field(default_factory=list)
    macro_switches: int = 0
    anti_repeat_count: int = 0
    avg_coherence: float = 0.0
    avg_macro_conf: float = 0.0
    total_intrinsic: float = 0.0
    ppo_entropy_avg: float = 0.0
    record_type: str = "episode"


class RunLogger:
    """Combined JSONL logger + live HUD printer.
    
    Creates a JSONL file at ``log_dir/<run_tag>.jsonl`` and writes
    structured records for offline analysis.  Also provides compact
    ``print_hud_line()`` / ``print_episode_summary()`` helpers that
    emit a single rich-formatted line to the terminal.
    """

    def __init__(self, run_tag: str, log_dir: str = "logs", hud_every: int = 1):
        self.run_tag = run_tag
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / f"{run_tag}.jsonl"
        self.hud_every = max(1, hud_every)
        self._step_count = 0
        self._episode_coherences: List[float] = []
        self._episode_macro_confs: List[float] = []
        self._episode_intrinsic: float = 0.0
        self._fh = open(self.jsonl_path, "a")
        logger.info(f"RunLogger: writing to {self.jsonl_path}")

    # ── JSONL writing ────────────────────────────────────────────────
    def _write(self, record: dict) -> None:
        record["ts"] = time.time()
        record["run"] = self.run_tag
        try:
            self._fh.write(json.dumps(record, default=str) + "\n")
            self._fh.flush()
        except Exception as e:
            logger.debug(f"JSONL write error: {e}")

    def log_step(self, rec: StepRecord) -> None:
        self._write(asdict(rec))
        self._step_count += 1
        self._episode_coherences.append(rec.coherence)
        self._episode_macro_confs.append(rec.macro_conf)
        self._episode_intrinsic += rec.intrinsic_reward

    def log_episode(self, summary: EpisodeSummary) -> None:
        self._write(asdict(summary))
        self._reset_episode_accumulators()

    def _reset_episode_accumulators(self) -> None:
        self._episode_coherences.clear()
        self._episode_macro_confs.clear()
        self._episode_intrinsic = 0.0

    # ── Live HUD ─────────────────────────────────────────────────────
    def print_hud_line(
        self,
        run_tag: str,
        ep: int,
        step: int,
        phase: str,
        macro: str,
        macro_conf: float,
        coherence: float,
        source: str,
        reward_delta: float,
        intrinsic: float,
        anti_repeat: bool,
        codex: bool,
        unique_cmds: int,
    ) -> None:
        """Print a compact one-line HUD to terminal.
        
        Format:
        [R66 ms2] EP3 s14 phase=PRIV_ESC macro=XYZ conf=0.72 coh=0.61 src=ppo r+=+12.4 rnd=+1.2 ar=0 codex=0 uniq=41
        """
        if self._step_count % self.hud_every != 0:
            return
        line = (
            f"[{run_tag}] EP{ep} s{step:02d} "
            f"phase={phase[:14]:14s} "
            f"macro={macro[:10]:10s} "
            f"conf={macro_conf:.2f} "
            f"coh={coherence:.2f} "
            f"src={source[:16]:16s} "
            f"r+={reward_delta:+7.1f} "
            f"rnd={intrinsic:+5.2f} "
            f"ar={'1' if anti_repeat else '0'} "
            f"cdx={'1' if codex else '0'} "
            f"uniq={unique_cmds}"
        )
        print(line, flush=True)

    def print_episode_summary(
        self,
        run_tag: str,
        ep: int,
        total_reward: float,
        steps: int,
        highest_phase: str,
        sources: Dict[str, int],
        discoveries: int,
        unique_cmds: int,
        anti_repeat: int,
        codex_calls: int,
        macro_switches: int,
        avg_coherence: float,
        avg_macro_conf: float,
    ) -> None:
        """Print compact episode summary block."""
        total = max(sum(sources.values()), 1)
        ppo_pct = sources.get("ppo", 0) / total * 100
        lines = [
            f"╭── EP{ep} SUMMARY {'─' * 50}",
            f"│ reward={total_reward:+.1f}  steps={steps}  phase={highest_phase}",
            f"│ ppo={ppo_pct:.0f}%  ar={anti_repeat}  codex={codex_calls}  "
            f"macro_sw={macro_switches}  disc={discoveries}  uniq={unique_cmds}",
            f"│ coherence={avg_coherence:.2f}  macro_conf={avg_macro_conf:.2f}",
            f"╰{'─' * 62}",
        ]
        for l in lines:
            print(l, flush=True)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
