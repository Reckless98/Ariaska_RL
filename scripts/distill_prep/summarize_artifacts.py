#!/usr/bin/env python3
"""Summarize distillation prep artifacts + generate weakness report.

Parses synthetic traces and teacher trajectories to produce:
  - Rich summary tables (phase histogram, tool coverage, reward stats)
  - Weakness report (curriculum/weakness_report.json)
  - Manifest with checksums (manifest.json)

Usage:
    python -m scripts.distill_prep.summarize_artifacts \\
        --traces-dir data/distill_prep/synthetic_traces \\
        --trajectories-dir data/distill_prep/teacher_trajectories
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ariaska.distill_prep.summarize_artifacts")


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def _parse_traces(traces_dir: Path) -> Dict[str, Any]:
    """Parse synthetic traces and collect statistics."""
    stats: Dict[str, Any] = {
        "total_runs": 0,
        "total_steps": 0,
        "phase_hist": Counter(),
        "tool_family_hist": Counter(),
        "decision_source_hist": Counter(),
        "reward_by_phase": defaultdict(list),
        "reward_by_tool": defaultdict(list),
        "discoveries_by_type": Counter(),
        "wrong_moves": 0,
        "difficulties": Counter(),
        "highest_phases": Counter(),
        "total_discoveries": 0,
        "command_repeats": Counter(),
    }

    if not traces_dir.exists():
        return stats

    for fp in sorted(traces_dir.glob("*.jsonl")):
        last_commands: List[str] = []
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                kind = obj.get("kind", "")
                if kind == "episode_start":
                    stats["total_runs"] += 1
                    stats["difficulties"][obj.get("difficulty", "unknown")] += 1
                    last_commands = []
                elif kind == "step":
                    stats["total_steps"] += 1
                    phase = obj.get("phase_before", "")
                    stats["phase_hist"][phase] += 1

                    for rec in obj.get("agent_records", []):
                        family = rec.get("command_family", "unknown")
                        stats["tool_family_hist"][family] += 1
                        source = rec.get("decision_source", "unknown")
                        stats["decision_source_hist"][source] += 1
                        reward = rec.get("reward", 0.0)
                        stats["reward_by_phase"][phase].append(reward)
                        stats["reward_by_tool"][family].append(reward)
                        if rec.get("is_wrong_move", False):
                            stats["wrong_moves"] += 1
                        cmd = rec.get("command", "")
                        if cmd in last_commands:
                            stats["command_repeats"][cmd] += 1
                        last_commands.append(cmd)
                        for disc in rec.get("discoveries", []):
                            dt = disc.get("discovery_type", "unknown")
                            stats["discoveries_by_type"][dt] += 1
                            stats["total_discoveries"] += 1

                elif kind == "episode_end":
                    hp = obj.get("highest_phase", "")
                    stats["highest_phases"][hp] += 1

    return stats


def _parse_trajectories(traj_dir: Path) -> Dict[str, Any]:
    """Parse teacher trajectories and collect statistics."""
    stats: Dict[str, Any] = {
        "total_trajectories": 0,
        "total_steps": 0,
        "phase_hist": Counter(),
        "tool_family_hist": Counter(),
        "wrong_moves": 0,
        "total_discoveries": 0,
        "discoveries_by_type": Counter(),
        "reward_by_phase": defaultdict(list),
        "difficulties": Counter(),
        "highest_phases": Counter(),
        "success_count": 0,
    }

    if not traj_dir.exists():
        return stats

    for fp in sorted(traj_dir.glob("*.jsonl")):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                kind = obj.get("kind", "")
                if kind == "trajectory_start":
                    stats["total_trajectories"] += 1
                    stats["difficulties"][obj.get("difficulty", "unknown")] += 1
                elif kind == "teacher_step":
                    stats["total_steps"] += 1
                    phase = obj.get("phase", "")
                    stats["phase_hist"][phase] += 1
                    family = obj.get("command_family", "unknown")
                    stats["tool_family_hist"][family] += 1
                    reward = obj.get("reward", 0.0)
                    stats["reward_by_phase"][phase].append(reward)
                    if obj.get("is_wrong_move", False):
                        stats["wrong_moves"] += 1
                    for disc in obj.get("discoveries", []):
                        dt = disc.get("discovery_type", "unknown")
                        stats["discoveries_by_type"][dt] += 1
                        stats["total_discoveries"] += 1
                elif kind == "trajectory_end":
                    hp = obj.get("highest_phase", "")
                    stats["highest_phases"][hp] += 1
                    if obj.get("success", False):
                        stats["success_count"] += 1

    return stats


# ---------------------------------------------------------------------------
# Weakness report
# ---------------------------------------------------------------------------


def generate_weakness_report(
    traces_dir: str = "data/distill_prep/synthetic_traces",
    trajectories_dir: str = "data/distill_prep/teacher_trajectories",
    output_path: str = "data/distill_prep/curriculum/weakness_report.json",
) -> Dict[str, Any]:
    """Generate weakness report from traces and trajectories."""
    from scripts.distill_prep.trace_schema import (
        COMMAND_FAMILIES,
        DISTILL_PREP_VERSION,
        PHASE_ORDER,
    )

    trace_stats = _parse_traces(Path(traces_dir))
    traj_stats = _parse_trajectories(Path(trajectories_dir))

    # Combine phase histograms
    combined_phase: Dict[str, int] = {}
    for phase in PHASE_ORDER:
        combined_phase[phase] = (
            trace_stats["phase_hist"].get(phase, 0)
            + traj_stats["phase_hist"].get(phase, 0)
        )

    # Tool family coverage
    combined_tools: Dict[str, int] = {}
    all_families = set(trace_stats["tool_family_hist"].keys()) | set(
        traj_stats["tool_family_hist"].keys()
    )
    for family in sorted(all_families):
        combined_tools[family] = (
            trace_stats["tool_family_hist"].get(family, 0)
            + traj_stats["tool_family_hist"].get(family, 0)
        )

    # Average rewards by phase
    avg_reward_phase: Dict[str, float] = {}
    for phase in PHASE_ORDER:
        rewards = (
            trace_stats["reward_by_phase"].get(phase, [])
            + traj_stats["reward_by_phase"].get(phase, [])
        )
        avg_reward_phase[phase] = round(sum(rewards) / max(len(rewards), 1), 3)

    # Average rewards by tool
    avg_reward_tool: Dict[str, float] = {}
    for family in sorted(all_families):
        rewards = (
            trace_stats["reward_by_tool"].get(family, [])
        )
        avg_reward_tool[family] = round(sum(rewards) / max(len(rewards), 1), 3)

    # Decision source percentages
    total_decisions = sum(trace_stats["decision_source_hist"].values()) or 1
    decision_pct: Dict[str, float] = {
        src: round(count / total_decisions * 100, 2)
        for src, count in trace_stats["decision_source_hist"].items()
    }

    # Repeated command patterns (top 10)
    repeated = [
        {"command": cmd, "count": count}
        for cmd, count in trace_stats["command_repeats"].most_common(10)
    ]

    # Weakness areas
    weaknesses: List[str] = []
    coverage_gaps: List[str] = []

    # Check underrepresented phases
    for phase in PHASE_ORDER:
        if combined_phase.get(phase, 0) < 10:
            weaknesses.append(f"Low coverage for phase {phase}: {combined_phase.get(phase, 0)} steps")

    # Check missing tool families
    for family in COMMAND_FAMILIES:
        if family not in combined_tools:
            coverage_gaps.append(f"No examples for tool family: {family}")

    # Check wrong move ratio
    total_steps = trace_stats["total_steps"] + traj_stats["total_steps"]
    total_wrong = trace_stats["wrong_moves"] + traj_stats["wrong_moves"]
    wrong_ratio = total_wrong / max(total_steps, 1)
    if wrong_ratio < 0.10:
        weaknesses.append(
            f"Wrong move ratio too low ({wrong_ratio:.1%}); target >= 15%"
        )
    elif wrong_ratio > 0.40:
        weaknesses.append(
            f"Wrong move ratio too high ({wrong_ratio:.1%}); may poison learner"
        )

    report = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "distill_prep_version": DISTILL_PREP_VERSION,
        "total_traces": trace_stats["total_runs"],
        "total_trajectories": traj_stats["total_trajectories"],
        "total_steps": total_steps,
        "phase_histogram": combined_phase,
        "repeated_command_patterns": repeated,
        "tool_family_coverage": combined_tools,
        "avg_reward_by_phase": avg_reward_phase,
        "avg_reward_by_tool": avg_reward_tool,
        "decision_source_pct": decision_pct,
        "weakness_areas": weaknesses,
        "coverage_gaps": coverage_gaps,
        "wrong_move_ratio": round(wrong_ratio, 4),
        "total_discoveries": trace_stats["total_discoveries"] + traj_stats["total_discoveries"],
        "discovery_distribution": dict(
            trace_stats["discoveries_by_type"] + traj_stats["discoveries_by_type"]
        ),
    }

    # Write
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote weakness report to %s", out)

    return report


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def generate_manifest(
    base_dir: str = "data/distill_prep",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate manifest with checksums for all artifacts."""
    from scripts.distill_prep.trace_schema import DISTILL_PREP_VERSION

    base = Path(base_dir)
    manifest: Dict[str, Any] = {
        "distill_prep_version": DISTILL_PREP_VERSION,
        "git_commit": _git_commit(),
        "seed": seed,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "counts": {},
        "files": [],
    }

    # Count and checksum all artifacts
    for subdir in ["synthetic_traces", "teacher_trajectories", "scenarios", "curriculum"]:
        dir_path = base / subdir
        if not dir_path.exists():
            manifest["counts"][subdir] = 0
            continue

        files = sorted(dir_path.glob("*"))
        files = [f for f in files if f.is_file()]
        manifest["counts"][subdir] = len(files)

        for fp in files:
            line_count = 0
            if fp.suffix in (".jsonl", ".json"):
                with open(fp, "r", encoding="utf-8") as f:
                    line_count = sum(1 for _ in f)

            manifest["files"].append(
                {
                    "path": str(fp.relative_to(base)),
                    "sha256": _sha256_file(fp),
                    "size_bytes": fp.stat().st_size,
                    "line_count": line_count,
                }
            )

    # Write
    manifest_path = base / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest to %s", manifest_path)

    return manifest


# ---------------------------------------------------------------------------
# Rich summary table
# ---------------------------------------------------------------------------


def print_summary(
    traces_dir: str = "data/distill_prep/synthetic_traces",
    trajectories_dir: str = "data/distill_prep/teacher_trajectories",
    weakness_report_path: str = "data/distill_prep/curriculum/weakness_report.json",
) -> None:
    """Print rich summary tables."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
    except ImportError:
        logger.info("Install 'rich' for formatted output")
        return

    console = Console()

    # Load weakness report
    report: Optional[Dict[str, Any]] = None
    wr_path = Path(weakness_report_path)
    if wr_path.exists():
        with open(wr_path, "r", encoding="utf-8") as f:
            report = json.load(f)

    if report is None:
        console.print("[yellow]No weakness report found. Run generation first.[/yellow]")
        return

    # Overview
    console.print(
        Panel(
            f"[bold]Distill Prep Summary[/bold]\n"
            f"Version: {report.get('distill_prep_version', '?')}\n"
            f"Generated: {report.get('generated_at', '?')}\n"
            f"Traces: {report.get('total_traces', 0)}  |  "
            f"Trajectories: {report.get('total_trajectories', 0)}  |  "
            f"Steps: {report.get('total_steps', 0)}  |  "
            f"Discoveries: {report.get('total_discoveries', 0)}",
            title="Overview",
        )
    )

    # Phase histogram
    phase_table = Table(title="Phase Distribution")
    phase_table.add_column("Phase", style="cyan")
    phase_table.add_column("Steps", justify="right")
    phase_table.add_column("Avg Reward", justify="right")
    for phase, count in report.get("phase_histogram", {}).items():
        avg_r = report.get("avg_reward_by_phase", {}).get(phase, 0.0)
        phase_table.add_row(phase, str(count), f"{avg_r:.2f}")
    console.print(phase_table)

    # Tool family coverage (top 20)
    tool_table = Table(title="Tool Family Coverage (Top 20)")
    tool_table.add_column("Family", style="cyan")
    tool_table.add_column("Count", justify="right")
    tool_table.add_column("Avg Reward", justify="right")
    sorted_tools = sorted(
        report.get("tool_family_coverage", {}).items(),
        key=lambda x: x[1],
        reverse=True,
    )[:20]
    for family, count in sorted_tools:
        avg_r = report.get("avg_reward_by_tool", {}).get(family, 0.0)
        tool_table.add_row(family, str(count), f"{avg_r:.2f}")
    console.print(tool_table)

    # Decision source distribution
    ds_table = Table(title="Decision Source Distribution")
    ds_table.add_column("Source", style="cyan")
    ds_table.add_column("Percentage", justify="right")
    for src, pct in sorted(
        report.get("decision_source_pct", {}).items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        ds_table.add_row(src, f"{pct:.1f}%")
    console.print(ds_table)

    # Weaknesses
    if report.get("weakness_areas"):
        console.print("\n[bold yellow]Weakness Areas:[/bold yellow]")
        for w in report["weakness_areas"]:
            console.print(f"  [yellow]• {w}[/yellow]")

    if report.get("coverage_gaps"):
        console.print("\n[bold red]Coverage Gaps:[/bold red]")
        for g in report["coverage_gaps"]:
            console.print(f"  [red]• {g}[/red]")

    # Wrong move ratio
    wr = report.get("wrong_move_ratio", 0)
    color = "green" if 0.10 <= wr <= 0.30 else "yellow"
    console.print(f"\n[{color}]Wrong move ratio: {wr:.1%}[/{color}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Summarize distill prep artifacts"
    )
    parser.add_argument(
        "--traces-dir",
        type=str,
        default="data/distill_prep/synthetic_traces",
    )
    parser.add_argument(
        "--trajectories-dir",
        type=str,
        default="data/distill_prep/teacher_trajectories",
    )
    parser.add_argument(
        "--weakness-report",
        type=str,
        default="data/distill_prep/curriculum/weakness_report.json",
    )
    parser.add_argument(
        "--manifest",
        action="store_true",
        help="Also generate manifest",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

    # Generate weakness report
    generate_weakness_report(
        args.traces_dir, args.trajectories_dir, args.weakness_report
    )

    # Optionally generate manifest
    if args.manifest:
        generate_manifest(seed=args.seed)

    # Print summary
    print_summary(args.traces_dir, args.trajectories_dir, args.weakness_report)


if __name__ == "__main__":
    main()
