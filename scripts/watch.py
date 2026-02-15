#!/usr/bin/env python3
"""
scripts/watch.py — Phase 9.7: Live training telemetry watcher

Tail and format JSONL telemetry files in real-time with Rich output.
Filters by agent, source, phase, or shows only discoveries/rewards.

Usage:
    python scripts/watch.py                           # Watch latest telemetry file
    python scripts/watch.py --file logs/telemetry/run_abc_telemetry.jsonl
    python scripts/watch.py --agent RedAgent           # Filter by agent
    python scripts/watch.py --source ppo               # Filter by decision source
    python scripts/watch.py --discoveries-only         # Show only steps with discoveries
    python scripts/watch.py --rewards-only             # Show only steps with reward > 0
    python scripts/watch.py --episode 5                # Filter by episode
    python scripts/watch.py --phase EXPLOITATION       # Filter by phase
"""

import argparse
import glob
import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


console = Console() if HAS_RICH else None


def find_latest_telemetry(log_dir: str = "logs/telemetry") -> str:
    """Find the most recently modified telemetry JSONL file."""
    pattern = os.path.join(log_dir, "*_telemetry.jsonl")
    files = glob.glob(pattern)
    if not files:
        return ""
    return max(files, key=os.path.getmtime)


def format_step_plain(record: dict) -> str:
    """Plain-text format for a step record."""
    agent = record.get("agent", "?")
    step = record.get("step", 0)
    ep = record.get("episode_id", 0)
    phase = record.get("phase", "?")
    source = record.get("source", "?")
    template = record.get("selected_template", "?")
    reward = record.get("reward_total", 0.0)
    disc_count = record.get("discovery_count", 0)
    ar = record.get("anti_repeat", {})
    ar_tag = " [AR]" if ar.get("triggered") else ""
    return (
        f"[E{ep:02d} S{step:02d}] {agent:<12s} | {phase:<16s} | "
        f"src={source:<12s} | {template:<24s} | "
        f"R={reward:+7.1f} | disc={disc_count}{ar_tag}"
    )


def format_episode_plain(record: dict) -> str:
    """Plain-text format for an episode summary."""
    ep = record.get("episode_id", 0)
    reward = record.get("total_reward", 0.0)
    steps = record.get("total_steps", 0)
    phase = record.get("final_phase", "?")
    disc = record.get("total_discoveries", 0)
    ar_pct = record.get("anti_repeat_pct", 0.0)
    return (
        f"═══ EPISODE {ep} ═══ reward={reward:+.1f} steps={steps} "
        f"phase={phase} discoveries={disc} anti_repeat={ar_pct:.0f}%"
    )


def format_step_rich(record: dict) -> None:
    """Rich-formatted step display."""
    if not HAS_RICH:
        print(format_step_plain(record))
        return
    
    agent = record.get("agent", "?")
    step = record.get("step", 0)
    ep = record.get("episode_id", 0)
    phase = record.get("phase", "?")
    source = record.get("source", "?")
    template = record.get("selected_template", "?")
    command = record.get("selected_command", "")[:60]
    reward = record.get("reward_total", 0.0)
    disc = record.get("discoveries", [])
    ar = record.get("anti_repeat", {})

    reward_color = "green" if reward > 0 else "red" if reward < 0 else "dim"
    ar_tag = " [yellow]⟳AR[/yellow]" if ar.get("triggered") else ""
    disc_str = ", ".join(f"{d['type']}:{d['value']}" for d in disc[:3]) if disc else ""

    console.print(
        f"[dim]E{ep:02d}[/dim] [cyan]S{step:02d}[/cyan] "
        f"[bold]{agent:<12s}[/bold] "
        f"[magenta]{phase:<12s}[/magenta] "
        f"[blue]{source:<10s}[/blue] "
        f"{template:<20s} "
        f"[{reward_color}]{reward:+7.1f}[/{reward_color}]"
        f"{ar_tag}"
        f"{'  🔍 ' + disc_str if disc_str else ''}"
    )


def format_episode_rich(record: dict) -> None:
    """Rich-formatted episode summary."""
    if not HAS_RICH:
        print(format_episode_plain(record))
        return

    ep = record.get("episode_id", 0)
    reward = record.get("total_reward", 0.0)
    steps = record.get("total_steps", 0)
    phase = record.get("final_phase", "?")
    disc = record.get("total_discoveries", 0)
    sources = record.get("source_distribution", {})
    ar_pct = record.get("anti_repeat_pct", 0.0)

    reward_color = "green" if reward > 500 else "yellow" if reward > 0 else "red"
    src_str = " ".join(f"{k}:{v}" for k, v in sorted(sources.items(), key=lambda x: -x[1])[:4])

    console.print(
        f"\n[bold]{'═' * 60}[/bold]"
    )
    console.print(
        f"[bold]EPISODE {ep}[/bold] │ "
        f"[{reward_color}]R={reward:+.1f}[/{reward_color}] │ "
        f"steps={steps} │ phase=[bold]{phase}[/bold] │ "
        f"disc={disc} │ AR={ar_pct:.0f}%"
    )
    if src_str:
        console.print(f"  [dim]sources: {src_str}[/dim]")
    console.print(f"[bold]{'═' * 60}[/bold]\n")


def should_display(record: dict, args) -> bool:
    """Apply user filters."""
    if record.get("type") == "episode_summary":
        if args.episode is not None and record.get("episode_id") != args.episode:
            return False
        return True

    if args.agent and record.get("agent") != args.agent:
        return False
    if args.source and record.get("source") != args.source:
        return False
    if args.phase and record.get("phase") != args.phase:
        return False
    if args.episode is not None and record.get("episode_id") != args.episode:
        return False
    if args.discoveries_only and not record.get("discoveries"):
        return False
    if args.rewards_only and record.get("reward_total", 0.0) <= 0:
        return False
    return True


def tail_file(filepath: str, args):
    """Tail a JSONL file, printing new lines as they appear."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    name = os.path.basename(filepath)
    if HAS_RICH:
        console.print(f"[bold cyan]Watching:[/bold cyan] {name}")
        console.print(f"[dim]Ctrl+C to stop[/dim]\n")
    else:
        print(f"Watching: {name}")
        print("Ctrl+C to stop\n")

    # If --follow, start from end; otherwise print everything
    with open(filepath, "r", encoding="utf-8") as f:
        if args.follow:
            f.seek(0, 2)  # Seek to end
        
        try:
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.2)
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                if not should_display(record, args):
                    continue

                if record.get("type") == "episode_summary":
                    if HAS_RICH:
                        format_episode_rich(record)
                    else:
                        print(format_episode_plain(record))
                else:
                    if HAS_RICH:
                        format_step_rich(record)
                    else:
                        print(format_step_plain(record))

        except KeyboardInterrupt:
            if HAS_RICH:
                console.print("\n[dim]Stopped.[/dim]")
            else:
                print("\nStopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Ariaska training telemetry watcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--file", "-f", help="Path to JSONL telemetry file")
    parser.add_argument("--agent", "-a", help="Filter by agent name (e.g. RedAgent)")
    parser.add_argument("--source", "-s", help="Filter by decision source (e.g. ppo, registry)")
    parser.add_argument("--phase", "-p", help="Filter by attack phase (e.g. EXPLOITATION)")
    parser.add_argument("--episode", "-e", type=int, help="Filter by episode number")
    parser.add_argument("--discoveries-only", "-d", action="store_true", help="Show only steps with discoveries")
    parser.add_argument("--rewards-only", "-r", action="store_true", help="Show only positive-reward steps")
    parser.add_argument("--follow", "-F", action="store_true", help="Start from end of file (like tail -f)")
    parser.add_argument("--no-color", action="store_true", help="Disable Rich formatting")

    args = parser.parse_args()

    if args.no_color:
        global HAS_RICH
        HAS_RICH = False

    filepath = args.file
    if not filepath:
        filepath = find_latest_telemetry()
        if not filepath:
            print("No telemetry files found in logs/telemetry/")
            print("Run a training session first, or specify --file path/to/file.jsonl")
            sys.exit(1)

    tail_file(filepath, args)


if __name__ == "__main__":
    main()
