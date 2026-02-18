"""
Episode Replayer — Phase 31
============================
Reads JSONL event trace files and renders a step-by-step Rich replay
of a past engagement. Used via CLI: ``ariaska replay <trace_file>``.

Author: Filip Volf
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

logger = logging.getLogger("ariaska.replay")

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ReplayAgentRecord:
    """Single agent action within a step."""
    agent_name: str = ""
    role: str = ""
    decision_source: str = ""
    phase: str = ""
    command: str = ""
    command_family: str = ""
    reward: float = 0.0
    mentor_call: bool = False
    discoveries: List[str] = field(default_factory=list)
    stdout_snippet: str = ""
    confidence: float = 0.0


@dataclass
class ReplayStep:
    """One step of a replayed episode."""
    step_num: int = 0
    phase_before: str = ""
    phase_after: str = ""
    step_reward_total: float = 0.0
    episode_reward_so_far: float = 0.0
    agent_records: List[ReplayAgentRecord] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class ReplayEpisode:
    """Full episode from a trace file."""
    episode_id: str = ""
    episode_num: int = 0
    steps: List[ReplayStep] = field(default_factory=list)
    total_reward: float = 0.0
    highest_phase: str = ""
    total_steps: int = 0
    target_ip: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────


def parse_trace_file(path: str) -> List[ReplayEpisode]:
    """
    Parse a JSONL event trace file into ReplayEpisode objects.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of ReplayEpisode objects (usually one per trace file).
    """
    episodes: Dict[str, ReplayEpisode] = {}
    trace_path = Path(path)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    with open(trace_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed JSON at line {line_num}")
                continue

            kind = event.get("kind", "")
            ep_id = event.get("episode_id", "unknown")

            if ep_id not in episodes:
                episodes[ep_id] = ReplayEpisode(episode_id=ep_id)

            ep = episodes[ep_id]

            if kind == "episode_start":
                ep.episode_num = event.get("episode_num", 0)
                data = event.get("data", {})
                ep.total_steps = data.get("max_steps", 0)

            elif kind == "step":
                step = ReplayStep(
                    step_num=event.get("step_num", 0),
                    phase_before=event.get("phase_before", ""),
                    phase_after=event.get("phase_after", ""),
                    step_reward_total=event.get("step_reward_total", 0.0),
                    episode_reward_so_far=event.get("episode_reward_so_far", 0.0),
                    timestamp=event.get("timestamp", 0.0),
                )
                if event.get("target_ip"):
                    ep.target_ip = event["target_ip"]

                for ar in event.get("agent_records", []):
                    rec = ReplayAgentRecord(
                        agent_name=ar.get("agent_name", ""),
                        role=ar.get("role", ""),
                        decision_source=ar.get("decision_source", ""),
                        phase=ar.get("phase", ""),
                        command=ar.get("command", ""),
                        command_family=ar.get("command_family", ""),
                        reward=ar.get("reward", 0.0),
                        mentor_call=ar.get("mentor_call", False),
                        discoveries=ar.get("discoveries", []),
                        stdout_snippet=ar.get("stdout_snippet", ""),
                        confidence=ar.get("confidence", 0.0),
                    )
                    step.agent_records.append(rec)

                ep.steps.append(step)

            elif kind == "episode_end":
                data = event.get("data", {})
                ep.total_reward = data.get("total_reward", 0.0)
                ep.highest_phase = data.get("highest_phase", "")
                ep.total_steps = data.get("steps", len(ep.steps))

    return list(episodes.values())


# ─────────────────────────────────────────────────────────────────────────────
# Rich renderer
# ─────────────────────────────────────────────────────────────────────────────


def render_episode(ep: ReplayEpisode, console: Console, verbose: bool = False) -> None:
    """
    Render a ReplayEpisode to the terminal using Rich.

    Args:
        ep: Episode to render.
        console: Rich Console.
        verbose: If True, show stdout snippets.
    """
    # ── Header ──
    console.print(Panel(
        f"[bold cyan]Episode {ep.episode_num}[/bold cyan]  "
        f"ID: {ep.episode_id}  Target: {ep.target_ip or 'unknown'}",
        title="REPLAY",
        border_style="blue",
    ))

    # ── Step-by-step table ──
    for step in ep.steps:
        step_table = Table(
            title=f"Step {step.step_num}  |  {step.phase_before} -> {step.phase_after}  "
                  f"|  Reward: {step.step_reward_total:+.1f}  "
                  f"|  Cumulative: {step.episode_reward_so_far:+.1f}",
            box=box.SIMPLE_HEAVY,
            show_lines=True,
        )
        step_table.add_column("Agent", style="cyan", width=14)
        step_table.add_column("Source", style="yellow", width=16)
        step_table.add_column("Command", style="white", width=42)
        step_table.add_column("Reward", style="green", width=8, justify="right")
        step_table.add_column("Discoveries", style="magenta", width=30)

        for rec in step.agent_records:
            disc_str = ", ".join(rec.discoveries[:5]) if rec.discoveries else "-"
            if len(rec.discoveries) > 5:
                disc_str += f" (+{len(rec.discoveries) - 5})"
            step_table.add_row(
                rec.agent_name,
                rec.decision_source,
                rec.command[:80] + ("..." if len(rec.command) > 80 else ""),
                f"{rec.reward:+.1f}",
                disc_str,
            )

        console.print(step_table)

        if verbose:
            for rec in step.agent_records:
                if rec.stdout_snippet:
                    snippet = rec.stdout_snippet[:300]
                    console.print(Panel(
                        snippet,
                        title=f"{rec.agent_name} output",
                        border_style="dim",
                        width=100,
                    ))

    # ── Summary ──
    summary = Table(title="Episode Summary", box=box.ROUNDED)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="bold white")
    summary.add_row("Total Steps", str(ep.total_steps))
    summary.add_row("Total Reward", f"{ep.total_reward:+.1f}")
    summary.add_row("Highest Phase", ep.highest_phase)
    summary.add_row("Target", ep.target_ip or "unknown")

    # Decision source distribution
    sources: Dict[str, int] = {}
    all_disc: List[str] = []
    for s in ep.steps:
        for r in s.agent_records:
            sources[r.decision_source] = sources.get(r.decision_source, 0) + 1
            all_disc.extend(r.discoveries)

    if sources:
        src_str = ", ".join(f"{k}={v}" for k, v in sorted(sources.items(), key=lambda x: -x[1]))
        summary.add_row("Decision Sources", src_str)
    summary.add_row("Total Discoveries", str(len(all_disc)))

    console.print(summary)


def replay_trace_file(
    path: str,
    verbose: bool = False,
    console: Optional[Console] = None,
) -> List[ReplayEpisode]:
    """
    Parse and render all episodes from a trace file.

    Args:
        path: Path to the JSONL trace file.
        verbose: Show stdout snippets.
        console: Rich Console (created if None).

    Returns:
        Parsed episodes.
    """
    if console is None:
        console = Console(force_terminal=True)

    episodes = parse_trace_file(path)
    if not episodes:
        console.print("[yellow]No episodes found in trace file.[/yellow]")
        return []

    for ep in episodes:
        render_episode(ep, console, verbose=verbose)

    return episodes
