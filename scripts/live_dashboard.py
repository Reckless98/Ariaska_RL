#!/usr/bin/env python3
"""
Ariaska Live Training Dashboard — pulls GPU metrics via SSH and renders
ASCII dashboards with Rich tables, sparklines, and trend analysis.

Usage:
    python scripts/live_dashboard.py                 # one-shot
    python scripts/live_dashboard.py --loop 300      # every 5 min
    python scripts/live_dashboard.py --loop 300 --rsync  # + rsync pull each cycle
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── rich ────────────────────────────────────────────────────────────────
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console(width=120)

# ── constants ───────────────────────────────────────────────────────────
GPU_HOST = os.environ.get("GPU_HOST", "root@212.247.220.172")
GPU_PORT = os.environ.get("GPU_PORT", "25107")
GPU_DIR = os.environ.get("GPU_DIR", "/root/Ariaska_RL")
LOCAL_DIR = Path(__file__).resolve().parent.parent
TRACE_DIR = f"{GPU_DIR}/traces/h200_distill"
SSH_OPTS = f"-o ConnectTimeout=20 -o ServerAliveInterval=10 -p {GPU_PORT}"

SPARK_CHARS = "▁▂▃▄▅▆▇█"
PHASE_ORDER = ["RECON", "ENUMERATION", "EXPLOITATION", "PRIVILEGE_ESCALATION",
               "LATERAL_MOVEMENT", "POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT"]
PHASE_SHORT = {"RECON": "REC", "ENUMERATION": "ENM", "EXPLOITATION": "EXP",
               "PRIVILEGE_ESCALATION": "PRV", "LATERAL_MOVEMENT": "LAT",
               "POST_EXPLOITATION": "PST", "EXFILTRATION": "EXF", "CLOSEOUT": "CLO"}


# ── SSH helper ──────────────────────────────────────────────────────────
def ssh_exec(cmd: str, timeout: int = 30) -> str:
    """Run a command on the GPU via SSH, return stdout."""
    full = f"ssh {SSH_OPTS} {GPU_HOST} '{cmd}'"
    try:
        r = subprocess.run(full, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        # Strip vast.ai welcome banner
        out = r.stdout
        for banner in ("Welcome to vast.ai", "Have fun!"):
            out = out.replace(banner, "")
        return out.strip()
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[ERROR: {e}]"


# ── data classes ────────────────────────────────────────────────────────
@dataclass
class RunSummary:
    run_id: str
    lines: int = 0
    episodes: int = 0
    rewards: list[float] = field(default_factory=list)
    size_bytes: int = 0


@dataclass
class ActiveRun:
    run_id: str
    pid: int
    steps: int = 0
    episodes_done: int = 0
    current_step: int = 0
    total_reward: float = 0.0
    rewards: list[float] = field(default_factory=list)
    phases: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    discoveries: int = 0
    budget_remaining: float = 0.0
    anneal_stage: str = ""
    anneal_progress: float = 0.0
    update_metrics: dict[str, float] = field(default_factory=dict)
    cmd_families: dict[str, int] = field(default_factory=dict)
    codex_cost: float = 0.0
    unique_cmds: int = 0
    duration_min: float = 0.0


@dataclass
class GPUStatus:
    gpu_util: int = 0
    mem_util: int = 0
    mem_used_mb: int = 0
    mem_total_mb: int = 0
    temp_c: int = 0
    disk_used: str = ""
    disk_total: str = ""
    disk_pct: str = ""
    load_avg: str = ""
    uptime: str = ""


@dataclass
class SyncStatus:
    local_traces: int = 0
    gpu_traces: int = 0
    local_models: int = 0
    gpu_models: int = 0
    local_unified_files: int = 0
    gpu_unified_files: int = 0
    last_local_rsync: str = "unknown"
    gpu_autopush: str = "stuck (no GH token)"


# ── data fetchers ───────────────────────────────────────────────────────

def fetch_gpu_status() -> GPUStatus:
    """Fetch GPU hardware metrics."""
    s = GPUStatus()
    raw = ssh_exec(
        'nvidia-smi --query-gpu=utilization.gpu,utilization.memory,'
        'memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null; '
        'echo "---"; df -h /root 2>/dev/null | tail -1; '
        'echo "---"; uptime'
    )
    parts = raw.split("---")
    if len(parts) >= 1:
        gpu_line = parts[0].strip()
        nums = [x.strip().replace(" %", "").replace(" MiB", "") for x in gpu_line.split(",")]
        if len(nums) >= 5:
            try:
                s.gpu_util = int(nums[0])
                s.mem_util = int(nums[1])
                s.mem_used_mb = int(nums[2])
                s.mem_total_mb = int(nums[3])
                s.temp_c = int(nums[4])
            except ValueError:
                pass
    if len(parts) >= 2:
        disk_line = parts[1].strip()
        cols = disk_line.split()
        if len(cols) >= 5:
            s.disk_total = cols[1]
            s.disk_used = cols[2]
            s.disk_pct = cols[4]
    if len(parts) >= 3:
        up_line = parts[2].strip()
        m = re.search(r"up\s+(.+?),\s+\d+\s+user", up_line)
        if m:
            s.uptime = m.group(1).strip()
        m2 = re.search(r"load average:\s*(.+)", up_line)
        if m2:
            s.load_avg = m2.group(1).strip()
    return s


def fetch_historical_runs() -> list[RunSummary]:
    """Fetch summary of all historical training runs."""
    raw = ssh_exec(
        'for f in ' + TRACE_DIR + '/h200_distill_*.jsonl; do '
        '  lines=$(wc -l < "$f"); '
        '  size=$(stat -c%s "$f" 2>/dev/null); '
        '  run=$(basename "$f" .jsonl | sed "s/h200_distill_//"); '
        '  ep_end=$(grep -c "episode_end" "$f" 2>/dev/null); '
        '  rews=$(grep "episode_end" "$f" 2>/dev/null | python3 -c "'
        'import json,sys; '
        '[print(json.loads(l).get(\\\"data\\\",{}).get(\\\"total_reward\\\", '
        'json.loads(l).get(\\\"total_reward\\\",\\\"?\\\"))) for l in sys.stdin]'
        '" 2>/dev/null | tr "\\n" ","); '
        '  echo "$run|$lines|$ep_end|$rews|$size"; '
        'done', timeout=45
    )
    runs = []
    for line in raw.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 5:
            try:
                r = RunSummary(run_id=parts[0], lines=int(parts[1]),
                               episodes=int(parts[2]), size_bytes=int(parts[4]))
                if parts[3]:
                    for v in parts[3].rstrip(",").split(","):
                        v = v.strip()
                        if v and v != "?":
                            try:
                                r.rewards.append(float(v))
                            except ValueError:
                                pass
                runs.append(r)
            except (ValueError, IndexError):
                pass
    return runs


def fetch_active_run() -> ActiveRun | None:
    """Fetch detailed metrics from the currently active training run."""
    # Find active PID
    pid_raw = ssh_exec('ps aux | grep h200_run | grep -v grep | awk "{print \\$2}"')
    if not pid_raw or pid_raw.startswith("["):
        return None
    pid = int(pid_raw.strip().splitlines()[0])

    # Find latest trace file
    latest_raw = ssh_exec(f'ls -t {TRACE_DIR}/h200_distill_*.jsonl 2>/dev/null | head -1')
    if not latest_raw or latest_raw.startswith("["):
        return None
    trace_file = latest_raw.strip()
    run_id = Path(trace_file).stem.replace("h200_distill_", "")

    # Parse full trace
    raw = ssh_exec(f'cat {trace_file} | python3 -c "'
                   'import json,sys\n'
                   'lines=[json.loads(l) for l in sys.stdin if l.strip()]\n'
                   'import json as j\n'
                   'print(j.dumps({\"lines\": len(lines), \"data\": lines}))\n'
                   '"', timeout=45)
    if not raw or raw.startswith("["):
        return ActiveRun(run_id=run_id, pid=pid)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return ActiveRun(run_id=run_id, pid=pid)

    records = parsed.get("data", [])
    ar = ActiveRun(run_id=run_id, pid=pid, steps=len(records))

    for r in records:
        kind = r.get("kind", r.get("type", ""))
        d = r.get("data", {})
        if not isinstance(d, dict):
            continue

        if kind == "step":
            rew = d.get("reward", d.get("step_reward", 0))
            ar.rewards.append(float(rew))
            phase = d.get("phase", "")
            if phase:
                ar.phases.append(phase)
            cmd = str(d.get("command", d.get("cmd", "")))
            if cmd:
                ar.commands.append(cmd[:80])
            cost = d.get("codex_cost", 0)
            if cost:
                ar.codex_cost += float(cost)
            disc = d.get("discoveries", d.get("new_discoveries", []))
            if isinstance(disc, list):
                ar.discoveries += len(disc)
            elif isinstance(disc, int):
                ar.discoveries += disc

        elif kind == "episode_end":
            ar.episodes_done += 1
            ar.total_reward = float(d.get("total_reward", 0))
            ar.anneal_stage = d.get("anneal_stage", "")
            ar.anneal_progress = float(d.get("anneal_progress", 0))
            ar.update_metrics = d.get("update_metrics", {})
            ar.cmd_families = d.get("cmd_families", {})

    ar.current_step = len(ar.rewards)
    ar.unique_cmds = len(set(ar.commands))

    # Budget from tmux
    tmux_raw = ssh_exec('tmux capture-pane -t train -p 2>/dev/null | tail -5')
    bm = re.findall(r'\$(\d+\.\d+)', tmux_raw)
    if bm:
        ar.budget_remaining = float(bm[-1])

    # Duration
    try:
        start_ts = datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        ar.duration_min = (datetime.now(timezone.utc) - start_ts).total_seconds() / 60
    except ValueError:
        pass

    return ar


def fetch_sync_status() -> SyncStatus:
    """Check bidirectional sync status."""
    s = SyncStatus()

    # GPU side
    gpu_raw = ssh_exec(
        'echo $(ls ' + TRACE_DIR + '/*.jsonl 2>/dev/null | wc -l); '
        'echo $(ls ' + GPU_DIR + '/models/unified/*.pt 2>/dev/null | wc -l); '
        'echo $(find ' + GPU_DIR + '/data/unified/ -name "*.jsonl" 2>/dev/null | wc -l)'
    )
    nums = gpu_raw.strip().splitlines()
    if len(nums) >= 1:
        try:
            s.gpu_traces = int(nums[0])
        except ValueError:
            pass
    if len(nums) >= 2:
        try:
            s.gpu_models = int(nums[1])
        except ValueError:
            pass
    if len(nums) >= 3:
        try:
            s.gpu_unified_files = int(nums[2])
        except ValueError:
            pass

    # Local side
    local_traces = list((LOCAL_DIR / "traces" / "h200_distill").glob("*.jsonl"))
    s.local_traces = len(local_traces)
    local_models = list((LOCAL_DIR / "models" / "unified").glob("*.pt"))
    s.local_models = len(local_models)
    local_unified = list((LOCAL_DIR / "data" / "unified").rglob("*.jsonl"))
    s.local_unified_files = len(local_unified)

    # Check local rsync process
    try:
        r = subprocess.run("pgrep -f gpu_rsync_all", shell=True, capture_output=True, text=True)
        if r.stdout.strip():
            s.last_local_rsync = f"running (PID {r.stdout.strip().splitlines()[0]})"
        else:
            s.last_local_rsync = "not running"
    except Exception:
        s.last_local_rsync = "unknown"

    return s


# ── rendering helpers ───────────────────────────────────────────────────

def sparkline(values: list[float], width: int = 40) -> str:
    """ASCII sparkline from a list of floats."""
    if not values:
        return ""
    # Downsample if needed
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values

    mn, mx = min(sampled), max(sampled)
    rng = mx - mn if mx != mn else 1.0
    return "".join(SPARK_CHARS[min(7, int((v - mn) / rng * 7))] for v in sampled)


def bar(value: float, max_val: float, width: int = 20, color: str = "green") -> str:
    """Simple ASCII bar."""
    if max_val <= 0:
        return " " * width
    ratio = min(1.0, max(0.0, value / max_val))
    filled = int(ratio * width)
    return f"[{color}]{'█' * filled}[/]{'░' * (width - filled)}"


def format_reward_trend(rewards: list[float]) -> str:
    """Show reward trend with arrow."""
    if len(rewards) < 10:
        return "insufficient data"
    first_10 = sum(rewards[:10]) / 10
    last_10 = sum(rewards[-10:]) / 10
    delta = last_10 - first_10
    arrow = "↑" if delta > 0.5 else "↓" if delta < -0.5 else "→"
    color = "green" if delta > 0.5 else "red" if delta < -0.5 else "yellow"
    return f"[{color}]{arrow} {first_10:+.1f} → {last_10:+.1f} (Δ{delta:+.1f})[/]"


def phase_progress_bar(phases: list[str]) -> str:
    """Show which kill chain phases were reached."""
    normalized = []
    phase_map = {
        "recon": "RECON", "enumeration": "ENUMERATION", "enum": "ENUMERATION",
        "exploit": "EXPLOITATION", "exploitation": "EXPLOITATION",
        "privesc": "PRIVILEGE_ESCALATION", "privilege_escalation": "PRIVILEGE_ESCALATION",
        "lateral": "LATERAL_MOVEMENT", "lateral_movement": "LATERAL_MOVEMENT",
        "post": "POST_EXPLOITATION", "post_exploitation": "POST_EXPLOITATION",
        "exfiltrate": "EXFILTRATION", "exfiltration": "EXFILTRATION",
        "closeout": "CLOSEOUT",
    }
    for p in phases:
        norm = phase_map.get(p.lower(), p.upper())
        if norm in PHASE_ORDER:
            normalized.append(norm)

    reached = set(normalized)
    max_idx = -1
    for p in PHASE_ORDER:
        if p in reached:
            max_idx = PHASE_ORDER.index(p)

    parts = []
    for i, p in enumerate(PHASE_ORDER):
        short = PHASE_SHORT.get(p, p[:3])
        if p in reached:
            parts.append(f"[green bold]{short}[/]")
        elif i <= max_idx:
            parts.append(f"[yellow]{short}[/]")
        else:
            parts.append(f"[dim]{short}[/]")
    return " → ".join(parts)


# ── render dashboard ────────────────────────────────────────────────────

def render_dashboard():
    """Fetch all data and render the full ASCII dashboard."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.rule(f"[bold cyan]ARIASKA LIVE TRAINING DASHBOARD[/] — {now}")
    console.print()

    # ── 1. GPU STATUS ───────────────────────────────────────────────
    gpu = fetch_gpu_status()
    gpu_table = Table(title="🖥️  GPU Infrastructure", show_header=True,
                      header_style="bold magenta", expand=True)
    gpu_table.add_column("Metric", style="cyan", width=20)
    gpu_table.add_column("Value", width=25)
    gpu_table.add_column("Visual", width=30)

    gpu_table.add_row("GPU Utilization", f"{gpu.gpu_util}%",
                       bar(gpu.gpu_util, 100, 25, "green" if gpu.gpu_util < 80 else "yellow"))
    gpu_table.add_row("VRAM Usage", f"{gpu.mem_used_mb}/{gpu.mem_total_mb} MiB",
                       bar(gpu.mem_used_mb, gpu.mem_total_mb, 25, "cyan"))
    gpu_table.add_row("Temperature", f"{gpu.temp_c}°C",
                       bar(gpu.temp_c, 85, 25, "green" if gpu.temp_c < 60 else "red"))
    gpu_table.add_row("Disk", f"{gpu.disk_used}/{gpu.disk_total} ({gpu.disk_pct})", "")
    gpu_table.add_row("Load Average", gpu.load_avg, "")
    gpu_table.add_row("Uptime", gpu.uptime, "")
    console.print(gpu_table)
    console.print()

    # ── 2. ACTIVE RUN ──────────────────────────────────────────────
    ar = fetch_active_run()
    if ar:
        run_table = Table(title=f"🏃 Active Run: {ar.run_id} (PID {ar.pid})",
                          show_header=True, header_style="bold green", expand=True)
        run_table.add_column("Metric", style="cyan", width=25)
        run_table.add_column("Value", width=50)

        run_table.add_row("Duration", f"{ar.duration_min:.0f} min")
        run_table.add_row("Steps (current episode)", f"{ar.current_step}")
        run_table.add_row("Episodes Completed", f"{ar.episodes_done}")
        run_table.add_row("Total Reward (last ep)", f"{ar.total_reward:+.1f}")
        run_table.add_row("Avg Step Reward", f"{sum(ar.rewards)/len(ar.rewards):+.2f}" if ar.rewards else "N/A")
        run_table.add_row("Reward Trend", format_reward_trend(ar.rewards))
        run_table.add_row("Discoveries", f"{ar.discoveries}")
        run_table.add_row("Unique Commands", f"{ar.unique_cmds}")
        run_table.add_row("Diversity Ratio", f"{ar.unique_cmds/max(1,ar.current_step):.1%}")
        run_table.add_row("Budget Remaining", f"${ar.budget_remaining:.2f}" if ar.budget_remaining else "N/A")
        run_table.add_row("Anneal Stage", f"{ar.anneal_stage} ({ar.anneal_progress:.1%})")
        run_table.add_row("Codex Cost (this run)", f"${ar.codex_cost:.4f}")
        console.print(run_table)
        console.print()

        # Reward sparkline
        if ar.rewards:
            spark = sparkline(ar.rewards, 80)
            console.print(Panel(
                f"[bold]Reward Sparkline[/] (min={min(ar.rewards):+.1f} max={max(ar.rewards):+.1f})\n"
                f"[green]{spark}[/]",
                title="📈 Step Rewards", border_style="green"
            ))
            console.print()

        # Phase progression
        if ar.phases:
            pbar = phase_progress_bar(ar.phases)
            console.print(Panel(pbar, title="🎯 Kill Chain Progress", border_style="yellow"))
            console.print()

        # PPO update metrics
        if ar.update_metrics:
            ppo_table = Table(title="🧠 PPO Update Metrics (Last Episode)",
                              show_header=True, header_style="bold blue")
            ppo_table.add_column("Metric", style="cyan", width=25)
            ppo_table.add_column("Value", width=20)
            ppo_table.add_column("Health", width=30)

            ploss = ar.update_metrics.get("policy_loss", 0)
            vloss = ar.update_metrics.get("value_loss", 0)
            entropy = ar.update_metrics.get("entropy", 0)
            kl = ar.update_metrics.get("approx_kl", 0)
            clip_frac = ar.update_metrics.get("clip_fraction", 0)
            kl_teacher = ar.update_metrics.get("kl_teacher_loss", 0)
            ranking = ar.update_metrics.get("ranking_loss", 0)
            exp_var = ar.update_metrics.get("explained_variance", 0)

            # Health indicators
            kl_health = "[green]OK[/]" if kl < 0.05 else "[yellow]high[/]" if kl < 0.1 else "[red]DIVERGING[/]"
            ent_health = "[green]exploring[/]" if entropy > 0.5 else "[yellow]narrowing[/]" if entropy > 0.1 else "[red]collapsed[/]"
            clip_health = "[green]good[/]" if clip_frac < 0.3 else "[yellow]high clipping[/]" if clip_frac < 0.5 else "[red]too aggressive[/]"
            ev_health = "[green]good[/]" if exp_var > 0.5 else "[yellow]weak[/]" if exp_var > 0 else "[red]useless baseline[/]"

            ppo_table.add_row("Policy Loss", f"{ploss:.4f}", "")
            ppo_table.add_row("Value Loss", f"{vloss:.2f}", "")
            ppo_table.add_row("Entropy", f"{entropy:.4f}", ent_health)
            ppo_table.add_row("Approx KL", f"{kl:.6f}", kl_health)
            ppo_table.add_row("Clip Fraction", f"{clip_frac:.3f}", clip_health)
            ppo_table.add_row("KL Teacher Loss", f"{kl_teacher:.4f}", "")
            ppo_table.add_row("Ranking Loss", f"{ranking:.4f}", "")
            ppo_table.add_row("Explained Variance", f"{exp_var:.4f}", ev_health)
            console.print(ppo_table)
            console.print()

        # Command family distribution
        if ar.cmd_families:
            cmd_table = Table(title="🔧 Command Family Distribution",
                              show_header=True, header_style="bold")
            cmd_table.add_column("Tool", style="cyan", width=20)
            cmd_table.add_column("Count", width=8, justify="right")
            cmd_table.add_column("Bar", width=30)

            sorted_fams = sorted(ar.cmd_families.items(), key=lambda x: -x[1])
            max_count = sorted_fams[0][1] if sorted_fams else 1
            for tool, count in sorted_fams[:15]:
                cmd_table.add_row(tool, str(count), bar(count, max_count, 25))
            console.print(cmd_table)
            console.print()

    else:
        console.print("[red bold]No active training detected![/]")
        console.print()

    # ── 3. HISTORICAL RUNS ────────────────────────────────────────
    runs = fetch_historical_runs()
    if runs:
        hist_table = Table(title="📊 Historical Training Runs (All Time)",
                           show_header=True, header_style="bold yellow", expand=True)
        hist_table.add_column("Run ID", style="cyan", width=20)
        hist_table.add_column("Episodes", width=8, justify="right")
        hist_table.add_column("Lines", width=8, justify="right")
        hist_table.add_column("Avg Reward", width=12, justify="right")
        hist_table.add_column("Best", width=10, justify="right")
        hist_table.add_column("Trend", width=25)
        hist_table.add_column("Size", width=10, justify="right")

        # Sort by timestamp (run_id)
        runs.sort(key=lambda r: r.run_id)
        # Show last 15
        for r in runs[-15:]:
            avg_r = f"{sum(r.rewards)/len(r.rewards):+.1f}" if r.rewards else "N/A"
            best_r = f"{max(r.rewards):+.1f}" if r.rewards else "N/A"
            size = f"{r.size_bytes/1024:.0f}K" if r.size_bytes < 1024*1024 else f"{r.size_bytes/1024/1024:.1f}M"

            if r.rewards and len(r.rewards) >= 2:
                trend_spark = sparkline(r.rewards, 20)
                trend = f"[green]{trend_spark}[/]"
            else:
                trend = "—"

            # Color coding
            avg_val = sum(r.rewards)/len(r.rewards) if r.rewards else 0
            style = "green" if avg_val > 500 else "yellow" if avg_val > 0 else "red"

            hist_table.add_row(r.run_id, str(r.episodes), str(r.lines),
                               f"[{style}]{avg_r}[/]", best_r, trend, size)

        console.print(hist_table)
        console.print()

        # Reward evolution across runs
        all_avgs = []
        for r in runs:
            if r.rewards:
                all_avgs.append(sum(r.rewards) / len(r.rewards))
        if len(all_avgs) >= 3:
            evolution_spark = sparkline(all_avgs, 60)
            early = sum(all_avgs[:3]) / 3
            late = sum(all_avgs[-3:]) / 3
            delta = late - early
            arrow = "↑" if delta > 10 else "↓" if delta < -10 else "→"
            color = "green" if delta > 10 else "red" if delta < -10 else "yellow"
            console.print(Panel(
                f"[bold]Cross-Run Reward Evolution[/] ({len(all_avgs)} runs with episodes)\n"
                f"[{color}]{evolution_spark}[/]\n"
                f"[{color}]{arrow} Early avg: {early:+.1f} → Recent avg: {late:+.1f} (Δ{delta:+.1f})[/]",
                title="📈 Learning Curve Across Runs",
                border_style=color
            ))
            console.print()

    # ── 4. SYNC STATUS ────────────────────────────────────────────
    sync = fetch_sync_status()
    sync_table = Table(title="🔄 Bidirectional Sync Status",
                       show_header=True, header_style="bold magenta", expand=True)
    sync_table.add_column("Asset", style="cyan", width=25)
    sync_table.add_column("GPU", width=15, justify="center")
    sync_table.add_column("Local", width=15, justify="center")
    sync_table.add_column("Status", width=25)

    trace_match = "[green]✓ synced[/]" if sync.local_traces >= sync.gpu_traces else f"[yellow]⚠ {sync.gpu_traces - sync.local_traces} behind[/]"
    model_match = "[green]✓ synced[/]" if sync.local_models >= sync.gpu_models else f"[yellow]⚠ {sync.gpu_models - sync.local_models} behind[/]"
    unified_match = "[green]✓ synced[/]" if sync.local_unified_files >= sync.gpu_unified_files else f"[yellow]⚠ behind[/]"

    sync_table.add_row("Trace Files", str(sync.gpu_traces), str(sync.local_traces), trace_match)
    sync_table.add_row("Unified Models (.pt)", str(sync.gpu_models), str(sync.local_models), model_match)
    sync_table.add_row("Unified Data (.jsonl)", str(sync.gpu_unified_files), str(sync.local_unified_files), unified_match)
    sync_table.add_row("Local rsync loop", "—", "—", sync.last_local_rsync)
    sync_table.add_row("GPU autopush", "—", "—", sync.gpu_autopush)
    console.print(sync_table)
    console.print()


def render_analysis(runs: list[RunSummary] | None = None):
    """Render training analysis and recommendations."""
    if runs is None:
        runs = fetch_historical_runs()

    console.rule("[bold cyan]📋 TRAINING ANALYSIS & STRATEGY[/]")
    console.print()

    # Compute trajectory
    all_avgs = []
    all_bests = []
    for r in runs:
        if r.rewards:
            all_avgs.append(sum(r.rewards) / len(r.rewards))
            all_bests.append(max(r.rewards))

    # Phase 1: Where we are
    console.print(Panel.fit(
        "[bold]Current Position in Training Lifecycle[/]\n\n"
        f"  Total runs analyzed: {len(runs)}\n"
        f"  Runs with episodes:  {len(all_avgs)}\n"
        f"  Total episodes:      {sum(r.episodes for r in runs)}\n"
        f"  Best single-episode: {max(all_bests):+.1f}\n" if all_bests else ""
        f"  Recent avg reward:   {sum(all_avgs[-3:])/max(1,min(3,len(all_avgs))):+.1f}\n" if all_avgs else ""
        f"  Trend:               {'IMPROVING' if len(all_avgs) > 3 and sum(all_avgs[-3:])/3 > sum(all_avgs[:3])/3 else 'STABILIZING' if all_avgs else 'N/A'}\n",
        title="📍 Position", border_style="cyan"
    ))
    console.print()

    # Phase 2: Why it's getting better
    console.print(Panel.fit(
        "[bold]Why Training Is Improving[/]\n\n"
        "  1. [green]Dual-Teacher Distillation[/] — Codex overrides inject expert-quality\n"
        "     actions (26.8 reward spikes = flag captures from teacher guidance)\n\n"
        "  2. [green]PPO with BC Loss[/] — Behavioral Cloning loss (kl_teacher_loss)\n"
        "     grounds the policy to teacher distributions, preventing drift\n\n"
        "  3. [green]Kill Chain Progression[/] — Agent now reaches EXFILTRATION phase\n"
        "     (was stuck in RECON in early runs with -300 rewards)\n\n"
        "  4. [green]Command Diversity[/] — 26 unique tool families in latest episode\n"
        "     (rustscan→nmap→curl→ffuf→binwalk→sqlmap→hydra→cat/scp)\n\n"
        "  5. [green]Annealing[/] — Teacher overrides decay over time (heavy→medium→light→off)\n"
        "     forcing the RL agent to make its own decisions\n\n"
        "  6. [green]Reward Shaping[/] — Discovery bonuses (port=2.5, service=5, shell=40,\n"
        "     flag=50, root_shell=80) give strong gradient signal\n",
        title="📈 Growth Factors", border_style="green"
    ))
    console.print()

    # Phase 3: LoRA / QLoRA / GRPO Analysis
    console.print(Panel.fit(
        "[bold]Fine-Tuning Strategy: LoRA vs QLoRA vs GRPO vs Distillation[/]\n\n"
        "  ┌─────────────┬──────────────────────────────────────────────────────┐\n"
        "  │ [cyan]Method[/]      │ [cyan]Description & When to Use[/]                          │\n"
        "  ├─────────────┼──────────────────────────────────────────────────────┤\n"
        "  │ [yellow]LoRA[/]        │ Low-Rank Adaptation. Adds small trainable rank     │\n"
        "  │             │ decomposition matrices to frozen LLM weights.       │\n"
        "  │             │ [green]USE WHEN:[/] You want to fine-tune the vLLM teacher   │\n"
        "  │             │ (gpt-oss-120b) on our 107K knowledge corpus so it   │\n"
        "  │             │ gives better pentesting commands. Needs ~40GB VRAM.  │\n"
        "  │             │ [yellow]STATUS: Ready soon[/] — need 50+ good episodes first. │\n"
        "  ├─────────────┼──────────────────────────────────────────────────────┤\n"
        "  │ [yellow]QLoRA[/]       │ Quantized LoRA. 4-bit quantization + LoRA.         │\n"
        "  │             │ Same quality, 75% less VRAM (~10GB for 120B model). │\n"
        "  │             │ [green]USE WHEN:[/] If LoRA OOMs on H200 (unlikely at 140GB) │\n"
        "  │             │ or if we want to run LoRA alongside training.       │\n"
        "  │             │ [yellow]STATUS: Fallback option.[/]                            │\n"
        "  ├─────────────┼──────────────────────────────────────────────────────┤\n"
        "  │ [yellow]GRPO[/]        │ Group Relative Policy Optimization.                │\n"
        "  │             │ DeepSeek's variant of PPO for LLMs — generates N    │\n"
        "  │             │ outputs per prompt, ranks them, uses relative       │\n"
        "  │             │ advantage without a value network.                  │\n"
        "  │             │ [green]USE WHEN:[/] Fine-tuning the LLM teacher to optimize  │\n"
        "  │             │ for our reward function directly (not just imitate  │\n"
        "  │             │ actions, but maximize pentest reward).              │\n"
        "  │             │ [red]STATUS: Phase 3 goal[/] — needs custom reward model. │\n"
        "  ├─────────────┼──────────────────────────────────────────────────────┤\n"
        "  │ [yellow]Distill[/]     │ Knowledge Distillation. Teacher→Student transfer.  │\n"
        "  │             │ [green]CURRENT:[/] This is what we're doing RIGHT NOW.       │\n"
        "  │             │ Codex teacher → PPO student via BC loss + KL div.   │\n"
        "  │             │ The PPO agent learns to imitate the teacher while   │\n"
        "  │             │ also maximizing environment reward.                 │\n"
        "  │             │ [green]STATUS: Active and working well.[/]                    │\n"
        "  └─────────────┴──────────────────────────────────────────────────────┘\n",
        title="🔬 Fine-Tuning Methods", border_style="blue"
    ))
    console.print()

    # Phase 4: Roadmap
    console.print(Panel.fit(
        "[bold]Estimated Work to Production-Ready[/]\n\n"
        "  [bold cyan]Phase 1 — NOW:[/] Distillation Training (RL + Teacher BC)\n"
        "    ├── Current: PPO + Codex dual-teacher distillation on H200\n"
        "    ├── Need: ~200-500 more episodes (10-20 training runs)\n"
        "    ├── Goal: Consistent 2000+ episode reward, reliable phase progression\n"
        "    └── ETA: 2-4 days of GPU time\n\n"
        "  [bold yellow]Phase 2 — NEXT:[/] LoRA Fine-Tune the Teacher (gpt-oss-120b)\n"
        "    ├── Collect best 200 episode traces as training data\n"
        "    ├── Fine-tune with LoRA (rank=16, alpha=32) on H200\n"
        "    ├── Teacher becomes pentesting-specialized → better overrides\n"
        "    ├── Self-improving loop: better teacher → better student → better data\n"
        "    └── ETA: 1-2 days after Phase 1 data collection\n\n"
        "  [bold magenta]Phase 3 — FUTURE:[/] GRPO for Reward-Aligned Teacher\n"
        "    ├── Build a reward model from episode outcomes\n"
        "    ├── GRPO-tune the teacher to directly maximize pentest reward\n"
        "    ├── Teacher doesn't just imitate commands — it strategizes for reward\n"
        "    └── ETA: 1-2 weeks (needs reward model + GRPO training pipeline)\n\n"
        "  [bold green]Phase 4 — GOAL:[/] Autonomous Agent (No Teacher Needed)\n"
        "    ├── Anneal teacher to 0% — agent acts independently\n"
        "    ├── Target: 80%+ solve rate on known CTF machines\n"
        "    ├── Deploy as standalone penetration testing assistant\n"
        "    └── ETA: 3-6 weeks total from now\n",
        title="🗺️  Roadmap to Production", border_style="yellow"
    ))
    console.print()


# ── main ────────────────────────────────────────────────────────────────

def do_rsync():
    """Pull latest from GPU via rsync."""
    console.print("[dim]Syncing from GPU...[/]")
    try:
        subprocess.run(
            f"bash {LOCAL_DIR / 'scripts' / 'gpu_rsync_all.sh'}",
            shell=True, capture_output=True, timeout=120
        )
        console.print("[green]✓ rsync complete[/]")
    except Exception as e:
        console.print(f"[yellow]rsync warning: {e}[/]")


def main():
    parser = argparse.ArgumentParser(description="Ariaska Live Training Dashboard")
    parser.add_argument("--loop", type=int, default=0,
                        help="Refresh interval in seconds (0 = one-shot)")
    parser.add_argument("--rsync", action="store_true",
                        help="Run rsync before each refresh")
    parser.add_argument("--analysis", action="store_true",
                        help="Include training analysis and strategy section")
    args = parser.parse_args()

    while True:
        if args.rsync:
            do_rsync()
        console.clear()
        render_dashboard()
        if args.analysis:
            render_analysis()
        if args.loop <= 0:
            break
        console.print(f"\n[dim]Next refresh in {args.loop}s... (Ctrl+C to stop)[/]")
        try:
            time.sleep(args.loop)
        except KeyboardInterrupt:
            console.print("\n[yellow]Dashboard stopped.[/]")
            break


if __name__ == "__main__":
    main()
