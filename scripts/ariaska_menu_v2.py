#!/usr/bin/env python3
"""
Ariaska Command Center — Interactive Terminal Wizard
=====================================================
Sleek, modern terminal UI for configuring and launching Ariaska engagements.

Usage:
    python scripts/ariaska_menu.py
    make ariaska
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

console = Console(force_terminal=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Visual Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ACCENT = "bright_cyan"
DIM = "dim"
OK = "green"
WARN = "yellow"
ERR = "red"

BANNER = """[bold bright_red]
    ▄▀▄ █▀▄ ▀█▀ ▄▀▄ ▄▀▀ █▄▀ ▄▀▄
    █▀█ █▀▄  █  █▀█ ▄██ █ █ █▀█[/bold bright_red]
[bold white]    ╔══════════════════════════════════════╗
    ║  Autonomous RL Pentesting Platform   ║
    ╚══════════════════════════════════════╝[/bold white]
[dim]    v3.0 ─ 5 Agents ─ PPO Actor-Critic ─ 107K Knowledge[/dim]"""

# Step indicator characters
STEP_DONE = "[green]●[/green]"
STEP_ACTIVE = "[bright_cyan]◉[/bright_cyan]"
STEP_PENDING = "[dim]○[/dim]"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class LaunchConfig:
    """Assembled configuration from the wizard flow."""
    target: str = ""
    mode: str = "ctf"              # normal | ctf
    ctf_goal: str = "both"         # user | root | both
    execution: str = "live"        # simulated | live
    compute: str = "cloud"         # cpu | gpu | cloud | remote
    steps: int = 500
    episodes: int = 1
    seed: Optional[int] = None
    verbosity: str = "verbose"
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    custom_env: Dict[str, str] = field(default_factory=dict)


@dataclass
class SystemInfo:
    """Detected environment capabilities."""
    gpu: bool = False
    gpu_name: str = ""
    gpu_vram: float = 0.0
    cuda: bool = False
    openai_key: bool = False
    local_llm: bool = False
    local_llm_model: str = ""
    python: str = ""
    test_count: int = 0
    last_run: str = ""
    total_runs: int = 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# System Detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_system() -> SystemInfo:
    """Detect current environment capabilities."""
    info = SystemInfo(python=sys.version.split()[0])

    # GPU / CUDA
    try:
        import torch
        if torch.cuda.is_available():
            info.gpu = True
            info.gpu_name = torch.cuda.get_device_name(0)
            info.gpu_vram = round(
                torch.cuda.get_device_properties(0).total_mem / (1024**3), 1
            )
            info.cuda = True
    except ImportError:
        pass

    # OpenAI API key
    info.openai_key = bool(os.environ.get("OPENAI_API_KEY"))

    # Local LLM probe
    port = int(os.environ.get("ARIASKA_LOCAL_LLM_PORT", "8192"))
    try:
        import urllib.request
        resp = urllib.request.urlopen(
            f"http://127.0.0.1:{port}/v1/models", timeout=2
        )
        data = json.loads(resp.read())
        if data.get("data"):
            info.local_llm = True
            info.local_llm_model = data["data"][0].get("id", "unknown")
    except Exception:
        pass

    # Training history
    artifacts = _PROJECT_ROOT / "artifacts"
    runs = (
        sorted(artifacts.glob("*_results.json"), reverse=True)
        if artifacts.exists() else []
    )
    info.total_runs = len(runs)
    if runs:
        info.last_run = runs[0].stem.replace("engagement_", "").replace(
            "_results", ""
        )

    return info


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UI Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _status_badge(ok: bool, label_ok: str, label_fail: str) -> str:
    if ok:
        return f"[black on green] {label_ok} [/black on green]"
    return f"[white on red] {label_fail} [/white on red]"


def _kv(key: str, val: str) -> str:
    return f"[dim]{key}:[/dim] [white]{val}[/white]"


def _prompt_icon() -> str:
    return "[bold bright_cyan]▸[/bold bright_cyan]"


def show_header(sys_info: SystemInfo) -> None:
    """Render branded header with system status badges."""
    console.print(BANNER)

    badges: list[str] = []
    if sys_info.openai_key:
        badges.append("[black on green] API ✓ [/black on green]")
    else:
        badges.append("[white on red] NO API KEY [/white on red]")

    if sys_info.gpu:
        short_name = sys_info.gpu_name.split()[-1]
        badges.append(
            f"[black on green] GPU {short_name} {sys_info.gpu_vram}G [/black on green]"
        )
    else:
        badges.append("[black on yellow] CPU-only [/black on yellow]")

    if sys_info.local_llm:
        badges.append(
            f"[black on green] LLM: {sys_info.local_llm_model[:20]} [/black on green]"
        )

    if sys_info.cuda:
        badges.append("[black on green] CUDA [/black on green]")

    badges.append(f"[dim] Py {sys_info.python} [/dim]")

    if sys_info.total_runs > 0:
        badges.append(f"[dim] {sys_info.total_runs} past runs [/dim]")

    console.print("    " + " ".join(badges))
    console.print()


def show_main_menu() -> str:
    """Render the two-column main menu and return choice key."""
    grid = Table(show_header=False, box=None, padding=(0, 3), expand=False)
    grid.add_column("col1", width=42)
    grid.add_column("col2", width=42)

    left = [
        ("1", "Launch Engagement",
         "Full wizard: target → mode → flags → go"),
        ("2", "Quick CTF",
         "Enter IP and go — optimized for flags"),
        ("3", "Training History",
         "Browse past runs, metrics, artifacts"),
        ("4", "System Diagnostics",
         "Health checks, imports, disk usage"),
    ]
    right = [
        ("5", "GPU / LLM Manager",
         "Start/stop local LLM, nvidia-smi"),
        ("6", "Run Test Suite",
         "pytest: quick / full / coverage"),
        ("7", "Feature Flags",
         "View/toggle all 76 runtime flags"),
        ("q", "Exit", ""),
    ]

    for l_item, r_item in zip(left, right):
        l_text = (
            f"  [bold bright_cyan]{l_item[0]}[/bold bright_cyan]"
            f"  [white]{l_item[1]}[/white]\n"
            f"      [dim]{l_item[2]}[/dim]"
        )
        r_text = (
            f"  [bold bright_cyan]{r_item[0]}[/bold bright_cyan]"
            f"  [white]{r_item[1]}[/white]\n"
            f"      [dim]{r_item[2]}[/dim]"
        )
        grid.add_row(l_text, r_text)

    console.print(Panel(
        grid,
        title="[bold white] COMMAND CENTER [/bold white]",
        border_style=ACCENT,
        box=box.HEAVY,
        padding=(1, 2),
    ))

    return Prompt.ask(
        f"    {_prompt_icon()}",
        choices=["1", "2", "3", "4", "5", "6", "7", "q"],
        default="1",
        show_choices=False,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Launch Wizard — 6-Step Flow
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WIZARD_STEPS = ["Target", "Mode", "Execution", "Compute", "Params", "Flags"]


def _step_bar(current: int) -> str:
    """Visual step progress indicator."""
    parts: list[str] = []
    for i, name in enumerate(WIZARD_STEPS):
        if i < current:
            parts.append(f"{STEP_DONE} {name}")
        elif i == current:
            parts.append(
                f"{STEP_ACTIVE} [bold bright_cyan]{name}[/bold bright_cyan]"
            )
        else:
            parts.append(f"{STEP_PENDING} [dim]{name}[/dim]")
    return "  ─  ".join(parts)


def _pick(
    prompt_text: str,
    options: list[tuple[str, str, str]],
    default: str = "1",
) -> str:
    """Compact option picker. Returns chosen key."""
    for key, label, desc in options:
        marker = (
            f"[{ACCENT}]▸[/{ACCENT}]" if key == default else " "
        )
        console.print(
            f"    {marker} [bold yellow]{key}[/bold yellow]"
            f"  {label}  [dim]{desc}[/dim]"
        )
    console.print()
    keys = [o[0] for o in options]
    return Prompt.ask(
        f"    {_prompt_icon()} {prompt_text}",
        choices=keys,
        default=default,
        show_choices=False,
    )


def wizard_launch(sys_info: SystemInfo) -> Optional[LaunchConfig]:
    """Full 6-step launch wizard."""
    cfg = LaunchConfig()
    console.print()
    console.print(Rule("[bold white]Launch Wizard[/bold white]", style=ACCENT))

    # ── Step 1: Target ──────────────────────────────────────────────
    console.print(f"\n  {_step_bar(0)}\n")
    cfg.target = Prompt.ask(
        f"    {_prompt_icon()} Target IP or domain",
        default="10.10.10.10",
    )
    if not cfg.target or cfg.target.count(".") < 1:
        console.print(f"    [{WARN}]⚠ Invalid — using default[/{WARN}]")
        cfg.target = "10.10.10.10"

    # ── Step 2: Mode ────────────────────────────────────────────────
    console.print(f"\n  {_step_bar(1)}\n")
    mode = _pick("Mode", [
        ("1", "Normal", "Full kill chain → CLOSEOUT"),
        ("2", "CTF — User Flag", "Hunt user.txt only"),
        ("3", "CTF — Root Flag", "Hunt root.txt only"),
        ("4", "CTF — Both Flags", "Auto-close on user.txt + root.txt"),
    ], default="4")
    mode_map = {
        "1": ("normal", "both"),
        "2": ("ctf", "user"),
        "3": ("ctf", "root"),
        "4": ("ctf", "both"),
    }
    cfg.mode, cfg.ctf_goal = mode_map[mode]

    # ── Step 3: Execution ───────────────────────────────────────────
    console.print(f"\n  {_step_bar(2)}\n")
    exec_choice = _pick("Execution", [
        ("1", "Live (SSH)", "Real commands against the target"),
        ("2", "Simulated", "No real commands — safe for testing"),
    ], default="1")
    cfg.execution = "live" if exec_choice == "1" else "simulated"

    # ── Step 4: Compute ─────────────────────────────────────────────
    console.print(f"\n  {_step_bar(3)}\n")
    compute_opts: list[tuple[str, str, str]] = [
        ("1", "Cloud (OpenAI)",
         f"All LLM via API. "
         f"{'[green]Key ✓[/green]' if sys_info.openai_key else '[red]No key![/red]'}"),
    ]
    if sys_info.gpu:
        compute_opts.append((
            "2",
            f"Local GPU ({sys_info.gpu_name.split()[-1]})",
            f"{sys_info.gpu_vram}G — nano/mini local, codex via API",
        ))
    compute_opts.append(
        ("3", "CPU Only", "Offline / deterministic mode"),
    )
    if sys_info.openai_key:
        compute_opts.append(
            ("4", "Remote GPU (vast.ai)", "SSH to cloud GPU for compute"),
        )

    compute_default = (
        "1" if sys_info.openai_key
        else ("2" if sys_info.gpu else "3")
    )
    comp = _pick("Compute", compute_opts, default=compute_default)
    cfg.compute = {"1": "cloud", "2": "gpu", "3": "cpu", "4": "remote"}[comp]

    # ── Step 5: Parameters ──────────────────────────────────────────
    console.print(f"\n  {_step_bar(4)}\n")
    cfg.steps = IntPrompt.ask(
        f"    {_prompt_icon()} Max steps per episode", default=500
    )
    cfg.episodes = IntPrompt.ask(
        f"    {_prompt_icon()} Episode count", default=1
    )
    seed_str = Prompt.ask(
        f"    {_prompt_icon()} Seed (empty = random)", default=""
    )
    cfg.seed = int(seed_str) if seed_str.strip().isdigit() else None

    verb = _pick("Verbosity", [
        ("1", "Quiet", "Summary only"),
        ("2", "Standard", "Step summaries, no raw output"),
        ("3", "Verbose", "Full live dashboard panels"),
    ], default="3")
    cfg.verbosity = {"1": "quiet", "2": "standard", "3": "verbose"}[verb]

    # ── Step 6: Feature Flags ───────────────────────────────────────
    console.print(f"\n  {_step_bar(5)}\n")
    if Confirm.ask(
        f"    {_prompt_icon()} Configure feature flags?", default=False
    ):
        cfg.feature_flags = _flag_picker()
    else:
        console.print("    [dim]Using defaults — profile settings.[/dim]")

    # ── Confirmation ────────────────────────────────────────────────
    console.print()
    _show_summary(cfg)
    console.print()

    if not Confirm.ask(
        f"    [bold bright_cyan]▸ Launch?[/bold bright_cyan]", default=True
    ):
        console.print(f"    [{WARN}]Cancelled.[/{WARN}]")
        return None

    return cfg


def _flag_picker() -> Dict[str, bool]:
    """Interactive feature flag toggle grid."""
    flag_groups: dict[str, list[tuple[str, str, bool]]] = {
        "RL & Training": [
            ("FF_BC_LOSS", "Behavioral Cloning loss", True),
            ("FF_TEACHER_TRACE", "Teacher distillation", True),
            ("FF_NEUROMODULATORS", "Bio-inspired RL control", True),
        ],
        "Intelligence": [
            ("FF_USE_MICRO_CHAIN", "MicroChain 3-stage", True),
            ("FF_EVIDENCE_GRAPH", "Evidence graph", True),
            ("FF_HYPOTHESIS_ENGINE", "Hypothesis engine", True),
        ],
        "Ops & Safety": [
            ("FF_OPS_HUB", "OpsHub pre-flight", True),
            ("FF_PHASE_INVARIANTS", "Phase hardening", True),
            ("FF_COMMAND_LOCKOUT", "Anti-repeat + cooldown", True),
            ("FF_SHELL_VALIDATOR", "Shell validation", True),
            ("FF_EXPLOIT_COOLDOWN", "Exploit cooldown", True),
        ],
        "Phase 39": [
            ("FF_ORION_RETHINK", "Orion deep-rethink", True),
            ("FF_TRUST_WEIGHTS", "Trust weight annealing", True),
            ("FF_CAP_GATE", "CAP regression gate", True),
            ("FF_DEBUG_TRACE", "Debug trace", True),
        ],
    }

    overrides: Dict[str, bool] = {}

    for group_name, flags in flag_groups.items():
        table = Table(
            show_header=False, box=None, padding=(0, 1),
            title=f"[bold]{group_name}[/bold]", title_style="dim",
        )
        table.add_column("", width=3)
        table.add_column("Flag", width=28)
        table.add_column("Description", width=28)
        table.add_column("Default", width=5)

        for env_key, desc, default in flags:
            s = f"[{OK}]ON[/{OK}]" if default else f"[{ERR}]OFF[/{ERR}]"
            table.add_row(
                "", f"[cyan]{env_key}[/cyan]", f"[dim]{desc}[/dim]", s
            )

        console.print(table)

    console.print()
    toggle_str = Prompt.ask(
        f"    {_prompt_icon()} Toggle flags "
        "(comma-separated, e.g. FF_BC_LOSS=0,FF_CAP_GATE=0)\n"
        "      [dim]Enter to keep defaults[/dim]",
        default="",
    )

    if toggle_str.strip():
        for pair in toggle_str.split(","):
            pair = pair.strip()
            if "=" in pair:
                k, v = pair.split("=", 1)
                overrides[k.strip()] = v.strip() not in (
                    "0", "false", "off", "no"
                )

    return overrides


def _show_summary(cfg: LaunchConfig) -> None:
    """Sleek config summary panel."""
    mode_display = cfg.mode.upper()
    if cfg.mode == "ctf":
        goal_map = {
            "user": "User Flag 🏴",
            "root": "Root Flag 🏴",
            "both": "Both Flags 🏴🏴",
        }
        mode_display = f"CTF → {goal_map[cfg.ctf_goal]}"

    exec_label = (
        "[bold]🔴 LIVE[/bold]"
        if cfg.execution == "live"
        else "[bold]🔵 SIMULATED[/bold]"
    )

    lines = [
        _kv("Target", f"[bold white]{cfg.target}[/bold white]"),
        _kv("Mode", mode_display),
        _kv("Execution", exec_label),
        _kv("Compute", cfg.compute.upper()),
        _kv("Steps",
            f"{cfg.steps} × {cfg.episodes} "
            f"episode{'s' if cfg.episodes > 1 else ''}"),
        _kv("Seed", str(cfg.seed) if cfg.seed else "random"),
        _kv("Verbosity", cfg.verbosity),
    ]
    if cfg.feature_flags:
        flag_str = ", ".join(
            f"{k}={'ON' if v else 'OFF'}"
            for k, v in cfg.feature_flags.items()
        )
        lines.append(_kv("Flags", flag_str))

    console.print(Panel(
        "\n".join(lines),
        title="[bold white] Configuration Summary [/bold white]",
        border_style=OK,
        box=box.ROUNDED,
        padding=(1, 3),
    ))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Quick CTF Mode
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def quick_ctf(sys_info: SystemInfo) -> Optional[LaunchConfig]:
    """Minimal-prompt CTF launch."""
    console.print()
    console.print(Rule(
        "[bold white]Quick CTF[/bold white]", style="bright_red"
    ))
    console.print(
        "    [dim]Optimized for flag hunting. Enter target and go.[/dim]\n"
    )

    target = Prompt.ask(
        f"    {_prompt_icon()} Target IP", default="10.10.10.10"
    )
    steps = IntPrompt.ask(
        f"    {_prompt_icon()} Max steps", default=500
    )
    seed_str = Prompt.ask(
        f"    {_prompt_icon()} Seed (empty = random)", default=""
    )
    seed = int(seed_str) if seed_str.strip().isdigit() else None

    cfg = LaunchConfig(
        target=target,
        mode="ctf",
        ctf_goal="both",
        execution="live",
        compute=(
            "cloud" if sys_info.openai_key
            else ("gpu" if sys_info.gpu else "cpu")
        ),
        steps=steps,
        episodes=1,
        seed=seed,
        verbosity="verbose",
    )

    _show_summary(cfg)
    console.print()
    if not Confirm.ask(
        f"    [bold bright_cyan]▸ Launch?[/bold bright_cyan]", default=True
    ):
        return None
    return cfg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Execution Engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_command(cfg: LaunchConfig) -> List[str]:
    """Assemble the CLI command."""
    cmd = [
        sys.executable, "ariaska_cli.py", "smart-train",
        "--target", cfg.target,
        "--steps", str(cfg.steps),
        "--verbosity", cfg.verbosity,
    ]
    if cfg.mode == "ctf":
        cmd.append("--ctf")
    if cfg.seed is not None:
        cmd.extend(["--seed", str(cfg.seed)])
    return cmd


def build_env(cfg: LaunchConfig) -> Dict[str, str]:
    """Build environment variables for the child process."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_PROJECT_ROOT)

    if cfg.execution == "simulated":
        env["ARIASKA_DRY_RUN"] = "1"

    if cfg.compute == "gpu":
        env["FF_LOCAL_LLM"] = "1"
        env["FF_LOCAL_LLM_OFFLOAD_NANO"] = "1"
        env["FF_LOCAL_LLM_OFFLOAD_MINI"] = "1"
    elif cfg.compute == "cpu":
        env["FF_LOCAL_LLM"] = "0"

    for k, v in cfg.feature_flags.items():
        env[k] = "1" if v else "0"

    for k, v in cfg.custom_env.items():
        env[k] = v

    return env


def execute(cfg: LaunchConfig, sys_info: SystemInfo) -> None:
    """Launch the engagement."""
    cmd = build_command(cfg)
    env = build_env(cfg)

    # Run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"ctf_{timestamp}_{cfg.target}"
        if cfg.mode == "ctf"
        else f"run_{timestamp}_{cfg.target}"
    )
    run_dir = _PROJECT_ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "live_output.log"

    # Persist config
    config_path = run_dir / "config.json"
    import dataclasses
    with open(config_path, "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2, default=str)

    console.print()
    console.print(Panel(
        f"[bold]Command:[/bold]  [dim]{' '.join(cmd)}[/dim]\n"
        f"[bold]Log:[/bold]      [dim]{log_path}[/dim]\n"
        f"[bold]Config:[/bold]   [dim]{config_path}[/dim]",
        title=f"[bold {ACCENT}] ▶ Launching [/bold {ACCENT}]",
        border_style=ACCENT,
        box=box.HEAVY,
        padding=(1, 2),
    ))
    console.print()

    t0 = time.time()

    if cfg.compute == "remote":
        console.print(
            f"    [{WARN}]Remote GPU — ensure vast.ai SSH is configured[/{WARN}]"
        )

    try:
        os.chdir(str(_PROJECT_ROOT))
        full_cmd = " ".join(cmd) + f" 2>&1 | tee {log_path}"
        os.execve("/bin/sh", ["/bin/sh", "-c", full_cmd], env)
    except Exception as e:
        console.print(
            f"    [{ERR}]execve failed: {e} — subprocess fallback[/{ERR}]"
        )
        try:
            with open(log_path, "w") as log_f:
                proc = subprocess.Popen(
                    cmd, env=env, cwd=str(_PROJECT_ROOT),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    sys.stdout.write(line)
                    log_f.write(line)
                proc.wait()
        except KeyboardInterrupt:
            console.print(f"\n    [{WARN}]Interrupted.[/{WARN}]")

    elapsed = time.time() - t0

    # Post-run sync
    console.print()
    console.print(Rule("[bold]Post-Run Artifact Sync[/bold]", style=OK))
    _sync_artifacts(run_dir, elapsed)


def _sync_artifacts(run_dir: Path, elapsed: float) -> None:
    """Copy results and traces into the run directory."""
    artifacts_dir = _PROJECT_ROOT / "artifacts"
    results = (
        sorted(artifacts_dir.glob("*_results.json"), reverse=True)
        if artifacts_dir.exists() else []
    )

    if results:
        latest = results[0]
        try:
            import shutil
            shutil.copy2(latest, run_dir / "results.json")
            data = json.loads(latest.read_text())

            uf = data.get("user_flag_captured", False)
            rf = data.get("root_flag_captured", False)
            steps = data.get("total_steps", 0)
            reward = data.get("total_reward", 0)
            phase = data.get("highest_phase", "RECON")

            status = (
                "[bold green]PWNED[/bold green]"
                if uf and rf
                else f"[{WARN}]INCOMPLETE[/{WARN}]"
            )

            console.print(f"    Result:    {status}")
            console.print(f"    Steps:     {steps}")
            console.print(f"    Reward:    {reward:+.1f}")
            console.print(f"    Phase:     {phase}")
            console.print(f"    Duration:  {elapsed:.0f}s")
            if uf:
                console.print(
                    f"    🏴 user:   [{OK}]{data.get('user_flag_value', '?')}[/{OK}]"
                )
            if rf:
                console.print(
                    f"    🏴 root:   [{OK}]{data.get('root_flag_value', '?')}[/{OK}]"
                )
        except Exception:
            pass

    # Copy latest trace
    traces_dir = _PROJECT_ROOT / "traces"
    if traces_dir.exists():
        traces = sorted(traces_dir.glob("events_*.jsonl"), reverse=True)
        if traces:
            import shutil
            shutil.copy2(traces[0], run_dir / "trace.jsonl")
            console.print(
                f"    Trace:     [dim]{traces[0].name} → {run_dir.name}/[/dim]"
            )

    console.print(f"    Artifacts: [dim]{run_dir}[/dim]")
    console.print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training History
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def show_history() -> None:
    """Browse past training runs with metrics."""
    console.print()
    console.print(Rule(
        "[bold white]Training History[/bold white]", style="bright_yellow"
    ))

    artifacts_dir = _PROJECT_ROOT / "artifacts"
    runs_dir = _PROJECT_ROOT / "runs"

    result_files = (
        sorted(artifacts_dir.glob("*_results.json"), reverse=True)
        if artifacts_dir.exists() else []
    )
    run_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir()]
        if runs_dir.exists() else [],
        key=lambda d: d.name, reverse=True,
    )

    if not result_files and not run_dirs:
        console.print("    [dim]No training history found.[/dim]")
        return

    if result_files:
        table = Table(
            box=box.SIMPLE_HEAVY, border_style="yellow",
            title=f"[bold]{len(result_files)} Engagements[/bold]",
            title_style="bright_yellow",
            padding=(0, 1),
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Timestamp", style="white", width=16)
        table.add_column("Target", style="cyan", width=16)
        table.add_column("Steps", style="white", width=6, justify="right")
        table.add_column("Reward", style="yellow", width=8, justify="right")
        table.add_column("Phase", style="bright_magenta", width=16)
        table.add_column("Flags", style="green", width=10)

        for i, f in enumerate(result_files[:20], 1):
            try:
                data = json.loads(f.read_text())
                ts = data.get("timestamp", "")[:16]
                target = data.get("target_ip", "?")
                ep = data.get("episode_metrics", {})
                steps = str(
                    data.get("total_steps", ep.get("total_steps", "?"))
                )
                reward = data.get("total_reward", 0)
                reward_str = (
                    f"{reward:+.1f}"
                    if isinstance(reward, (int, float))
                    else str(reward)
                )
                phase = data.get(
                    "highest_phase", ep.get("highest_phase", "?")
                )
                uf = data.get("user_flag_captured", False)
                rf = data.get("root_flag_captured", False)
                if uf and rf:
                    flags_str = "[bold green]🏴🏴 PWN[/bold green]"
                elif uf:
                    flags_str = f"[{OK}]🏴 user[/{OK}]"
                elif rf:
                    flags_str = f"[{OK}]🏴 root[/{OK}]"
                else:
                    flags_str = "[dim]—[/dim]"

                table.add_row(
                    str(i), ts, target, steps, reward_str, phase, flags_str
                )
            except Exception:
                table.add_row(str(i), f.name[:16], *["?"] * 4, "?")

        console.print(table)

    if run_dirs:
        console.print()
        console.print(f"    [dim]{len(run_dirs)} run directories in runs/[/dim]")
        for d in run_dirs[:10]:
            n = len(list(d.iterdir()))
            console.print(
                f"    [dim]📁[/dim] {d.name}  [dim]({n} files)[/dim]"
            )

    # Cost rollup
    total_cost = 0.0
    for f in result_files:
        try:
            data = json.loads(f.read_text())
            total_cost += float(
                data.get("total_cost_usd", data.get("cost_usd", 0))
            )
        except Exception:
            pass
    if total_cost > 0:
        console.print(
            f"\n    [dim]Total API cost:[/dim] "
            f"[bold red]${total_cost:.2f}[/bold red]"
        )

    console.print()

    if result_files and Confirm.ask("    View details?", default=False):
        idx = IntPrompt.ask("    Which #?", default=1)
        if 1 <= idx <= len(result_files):
            data = json.loads(result_files[idx - 1].read_text())
            console.print_json(json.dumps(data, indent=2, default=str)[:6000])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# System Diagnostics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def show_diagnostics() -> None:
    """Comprehensive system health check."""
    console.print()
    console.print(Rule(
        "[bold white]System Diagnostics[/bold white]", style="bright_red"
    ))

    console.print("\n    [bold]Dependencies[/bold]")
    deps = [
        ("torch", "PyTorch"), ("numpy", "NumPy"), ("rich", "Rich"),
        ("openai", "OpenAI"), ("chromadb", "ChromaDB"),
        ("sentence_transformers", "SentenceTransformers"),
    ]
    for mod, name in deps:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            console.print(f"    [{OK}]✓[/{OK}]  {name:<24} {ver}")
        except ImportError:
            console.print(
                f"    [{ERR}]✗[/{ERR}]  {name:<24} [dim]not installed[/dim]"
            )

    console.print("\n    [bold]Core Modules[/bold]")
    core = [
        "core.gpt_manager", "core.feature_flags",
        "core.algorithms.ppo_agent", "core.models.state_encoder",
        "core.llm.budget_manager", "core.training.smart_coach",
        "core.orchestration.smart_orchestrator",
    ]
    for mod in core:
        try:
            __import__(mod)
            console.print(f"    [{OK}]✓[/{OK}]  {mod}")
        except Exception as e:
            console.print(f"    [{ERR}]✗[/{ERR}]  {mod}  [dim]{e}[/dim]")

    console.print("\n    [bold]Storage[/bold]")
    for d in [
        "data/", "models/", "artifacts/", "traces/", "runs/", "postmortems/"
    ]:
        p = _PROJECT_ROOT / d
        if p.exists():
            try:
                r = subprocess.run(
                    ["du", "-sh", str(p)],
                    capture_output=True, text=True, timeout=10,
                )
                size = r.stdout.split()[0] if r.stdout else "?"
                console.print(f"    {d:<20} {size}")
            except Exception:
                console.print(f"    {d:<20} ?")

    console.print("\n    [bold]Configuration[/bold]")
    env_ok = (_PROJECT_ROOT / ".env").exists()
    console.print(
        f"    .env               "
        f"{_status_badge(env_ok, 'Found', 'Missing')}"
    )
    ckpt = _PROJECT_ROOT / "models" / "enhanced" / "ppo_live_checkpoint.pt"
    if ckpt.exists():
        size_kb = ckpt.stat().st_size / 1024
        console.print(f"    PPO checkpoint     [{OK}]{size_kb:.0f} KB[/{OK}]")
    else:
        console.print(
            "    PPO checkpoint     [dim]None — created on first run[/dim]"
        )

    console.print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GPU / LLM Manager
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def show_gpu_manager(sys_info: SystemInfo) -> None:
    """GPU and local LLM management submenu."""
    while True:
        console.print()
        console.print(Rule(
            "[bold white]GPU / LLM Manager[/bold white]",
            style="bright_magenta",
        ))

        if sys_info.gpu:
            console.print(
                f"    GPU:       [{OK}]{sys_info.gpu_name} "
                f"({sys_info.gpu_vram}G)[/{OK}]"
            )
        else:
            console.print("    GPU:       [dim]Not available[/dim]")
        if sys_info.local_llm:
            console.print(
                f"    Local LLM: [{OK}]ONLINE — "
                f"{sys_info.local_llm_model}[/{OK}]"
            )
        else:
            console.print("    Local LLM: [dim]Offline[/dim]")
        console.print()

        choice = _pick("Select", [
            ("1", "Start Local LLM", ""),
            ("2", "Stop Local LLM", ""),
            ("3", "Health Check", ""),
            ("4", "nvidia-smi", ""),
            ("5", "Download Model", ""),
            ("b", "Back", ""),
        ], default="b")

        if choice == "b":
            break
        elif choice == "1":
            _start_llm()
        elif choice == "2":
            _stop_llm()
        elif choice == "3":
            _check_llm()
        elif choice == "4":
            _nvidia_smi()
        elif choice == "5":
            _download_model()

        # Refresh status
        refreshed = detect_system()
        sys_info.local_llm = refreshed.local_llm
        sys_info.local_llm_model = refreshed.local_llm_model


def _start_llm() -> None:
    try:
        from core.llm.local_llm_provider import get_local_llm_provider
        provider = get_local_llm_provider()
        if provider.is_available():
            console.print(f"    [{WARN}]Already running[/{WARN}]")
            return
        console.print(f"    [{ACCENT}]Starting...[/{ACCENT}]")
        if provider.start_server():
            console.print(f"    [{OK}]✓ Started[/{OK}]")
        else:
            console.print(f"    [{ERR}]✗ Failed[/{ERR}]")
    except Exception as e:
        console.print(f"    [{ERR}]{e}[/{ERR}]")


def _stop_llm() -> None:
    try:
        from core.llm.local_llm_provider import get_local_llm_provider
        get_local_llm_provider().stop_server()
        console.print(f"    [{OK}]✓ Stopped[/{OK}]")
    except Exception as e:
        console.print(f"    [{ERR}]{e}[/{ERR}]")


def _check_llm() -> None:
    port = int(os.environ.get("ARIASKA_LOCAL_LLM_PORT", "8192"))
    try:
        import urllib.request
        resp = urllib.request.urlopen(
            f"http://127.0.0.1:{port}/v1/models", timeout=5
        )
        models = json.loads(resp.read())
        console.print(f"    [{OK}]✓ Responding on :{port}[/{OK}]")
        for m in models.get("data", []):
            console.print(f"      Model: [bold]{m.get('id', '?')}[/bold]")

        console.print(f"    [{ACCENT}]Inference test...[/{ACCENT}]")
        req_data = json.dumps({
            "model": "local",
            "messages": [{"role": "user", "content": "Say 'ready'"}],
            "max_tokens": 5,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=req_data,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        resp = urllib.request.urlopen(req, timeout=30)
        ms = (time.time() - t0) * 1000
        result = json.loads(resp.read())
        content = result["choices"][0]["message"]["content"].strip()
        console.print(
            f"      Response: [bold]{content}[/bold] ({ms:.0f}ms)"
        )
    except Exception as e:
        console.print(f"    [{ERR}]✗ {e}[/{ERR}]")


def _nvidia_smi() -> None:
    try:
        r = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        if r.returncode == 0:
            console.print(Panel(
                r.stdout, title="nvidia-smi",
                border_style=OK, box=box.ROUNDED,
            ))
        else:
            console.print(f"    [{WARN}]Not available[/{WARN}]")
    except FileNotFoundError:
        console.print(f"    [{WARN}]nvidia-smi not found[/{WARN}]")


def _download_model() -> None:
    console.print()
    t = Table(show_header=True, box=box.SIMPLE, padding=(0, 1))
    t.add_column("#", style="yellow", width=3)
    t.add_column("Model", style="white")
    t.add_column("Size", style="dim")
    t.add_column("VRAM", style="dim")
    t.add_row("1", "Qwen3-32B Q4_K_M", "~18.5 GB", "~20 GB")
    t.add_row("2", "Qwen3-32B Q5_K_M", "~22 GB", "~24 GB")
    t.add_row("3", "Qwen3-14B Q5_K_M", "~10 GB", "~12 GB")
    t.add_row("4", "Qwen3-8B Q8_0", "~8 GB", "~10 GB")
    console.print(t)

    choice = Prompt.ask("    Model #", choices=["1", "2", "3", "4"],
                        default="1")
    info = {
        "1": ("Qwen/Qwen3-32B-Instruct-GGUF",
              "Qwen3-32B-Instruct-Q4_K_M.gguf"),
        "2": ("Qwen/Qwen3-32B-Instruct-GGUF",
              "Qwen3-32B-Instruct-Q5_K_M.gguf"),
        "3": ("Qwen/Qwen3-14B-Instruct-GGUF",
              "Qwen3-14B-Instruct-Q5_K_M.gguf"),
        "4": ("Qwen/Qwen3-8B-Instruct-GGUF",
              "Qwen3-8B-Instruct-Q8_0.gguf"),
    }
    repo, filename = info[choice]
    model_dir = Path(
        os.environ.get("ARIASKA_MODEL_DIR", str(_PROJECT_ROOT / "models"))
    )
    model_dir.mkdir(parents=True, exist_ok=True)

    target = model_dir / filename
    if target.exists():
        console.print(f"    [{WARN}]Exists: {target}[/{WARN}]")
        if not Confirm.ask("    Re-download?", default=False):
            return

    console.print(f"    [{ACCENT}]Downloading {filename}...[/{ACCENT}]")
    subprocess.run([
        sys.executable, "-m", "huggingface_hub", "download",
        repo, filename,
        "--local-dir", str(model_dir),
        "--local-dir-use-symlinks", "False",
    ])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test Suite
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_tests() -> None:
    """Run the test suite."""
    console.print()
    console.print(Rule(
        "[bold white]Test Suite[/bold white]", style="bright_green"
    ))

    mode = _pick("Mode", [
        ("1", "Quick", "Stop on first failure, short output"),
        ("2", "Full", "All tests, even on failures"),
        ("3", "Coverage", "Full + coverage report"),
        ("4", "Specific", "Specify test file or pattern"),
    ], default="1")

    base_env = dict(os.environ)
    base_env["ARIASKA_DRY_RUN"] = "1"
    base_env["PYTHONPATH"] = str(_PROJECT_ROOT)
    pytest_bin = str(_PROJECT_ROOT / ".venv/bin/pytest")
    ignore = "--ignore=tests/test_online_makes_mentor_calls.py"

    if mode == "1":
        cmd = [
            pytest_bin, "tests/", "-x", "--tb=short", "-q",
            "--timeout=120", ignore,
        ]
    elif mode == "2":
        cmd = [
            pytest_bin, "tests/", "--tb=short", "-q",
            "--timeout=120", ignore,
        ]
    elif mode == "3":
        cmd = [
            pytest_bin, "tests/", "--tb=short", "-q", "--timeout=120",
            "--cov=core", "--cov-report=term-missing", ignore,
        ]
    else:
        path = Prompt.ask("    Test file/pattern", default="tests/")
        cmd = [pytest_bin, path, "-v", "--tb=short", "--timeout=120"]

    console.print(f"\n    [dim]{' '.join(cmd)}[/dim]\n")
    subprocess.run(cmd, env=base_env, cwd=str(_PROJECT_ROOT))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Feature Flags Viewer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def show_feature_flags() -> None:
    """View all active feature flags with env override status."""
    console.print()
    console.print(Rule(
        "[bold white]Feature Flags[/bold white]", style="bright_magenta"
    ))

    try:
        from core.feature_flags import FeatureFlags
        flags = FeatureFlags()

        table = Table(
            box=box.SIMPLE_HEAVY, border_style="bright_magenta",
            padding=(0, 1),
        )
        table.add_column("Flag", style="cyan", width=35)
        table.add_column("Value", width=7, justify="center")
        table.add_column("Env Override", style="dim", width=20)

        for attr in sorted(dir(flags)):
            if attr.startswith("_") or attr == "profile":
                continue
            val = getattr(flags, attr, None)
            if isinstance(val, bool):
                val_str = (
                    f"[{OK}]ON[/{OK}]" if val else f"[{ERR}]OFF[/{ERR}]"
                )
                env_key = f"FF_{attr.upper()}"
                env_val = os.environ.get(env_key, "")
                override = (
                    f"[{WARN}]{env_val}[/{WARN}]"
                    if env_val else "[dim]—[/dim]"
                )
                table.add_row(attr, val_str, override)
            elif isinstance(val, (int, float, str)) and not callable(val):
                table.add_row(attr, f"[white]{val}[/white]", "[dim]—[/dim]")

        console.print(table)
        profile = getattr(flags, "profile", "unknown")
        console.print(f"\n    [dim]Profile: {profile}[/dim]")
        console.print(
            "    [dim]Override: set FF_<FLAG_NAME>=0|1 in env[/dim]"
        )
    except Exception as e:
        console.print(f"    [{ERR}]{e}[/{ERR}]")

    console.print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    """Main entry point."""
    console.clear()
    sys_info = detect_system()
    show_header(sys_info)

    while True:
        try:
            choice = show_main_menu()

            if choice == "q":
                console.print("\n    [bold red]Ariaska offline.[/bold red]\n")
                break
            elif choice == "1":
                cfg = wizard_launch(sys_info)
                if cfg:
                    execute(cfg, sys_info)
            elif choice == "2":
                cfg = quick_ctf(sys_info)
                if cfg:
                    execute(cfg, sys_info)
            elif choice == "3":
                show_history()
            elif choice == "4":
                show_diagnostics()
            elif choice == "5":
                show_gpu_manager(sys_info)
            elif choice == "6":
                run_tests()
            elif choice == "7":
                show_feature_flags()

        except KeyboardInterrupt:
            console.print(f"\n    [{WARN}]Interrupted — returning to menu[/{WARN}]")
            continue
        except EOFError:
            break


if __name__ == "__main__":
    main()
