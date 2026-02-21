#!/usr/bin/env python3
"""
Ariaska Command Center v3.0 — Interactive Terminal Wizard
==========================================================
Senior-grade, demo-ready terminal UI for configuring and launching
Ariaska autonomous penetration testing engagements.

Features:
  • Full-art ASCII banner with gradient styling
  • Rich system status panel with GPU/LLM/API detection
  • 6-step launch wizard with visual progress indicators
  • Quick CTF mode — 3 prompts to flag hunting
  • Advanced config — model routing, PPO params, budget, env vars
  • Training history browser with flag status + cost tracker
  • GPU / LLM manager with health checks and model downloads
  • Full test suite runner with coverage
  • Post-run artifact sync with results display

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

# ── Project root on sys.path ────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

# ═══════════════════════════════════════════════════════════════════════
#  Visual Constants & Branding
# ═══════════════════════════════════════════════════════════════════════

BANNER = r"""[bold red]
     █████╗ ██████╗ ██╗ █████╗ ███████╗██╗  ██╗ █████╗
    ██╔══██╗██╔══██╗██║██╔══██╗██╔════╝██║ ██╔╝██╔══██╗
    ███████║██████╔╝██║███████║███████╗█████╔╝ ███████║
    ██╔══██║██╔══██╗██║██╔══██║╚════██║██╔═██╗ ██╔══██║
    ██║  ██║██║  ██║██║██║  ██║███████║██║  ██╗██║  ██║
    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝
[/bold red][bold white]    ══════ Autonomous Multi-Agent RL Pentesting ══════[/bold white]
[dim]    Phase 43 • PPO Actor-Critic v3.0 • 5 Agents • 107K Knowledge Corpus[/dim]"""

# Color palette — consistent across all panels
C_ACCENT   = "bright_cyan"
C_OK       = "green"
C_WARN     = "yellow"
C_ERR      = "red"
C_DIM      = "dim"
C_TITLE    = "bold white"
C_KEY      = "bold yellow"
C_LABEL    = "white"
C_DESC     = "dim"

# Wizard step indicators
STEP_DONE   = "[green]●[/green]"
STEP_ACTIVE = "[bright_cyan]◉[/bright_cyan]"
STEP_PENDING = "[dim]○[/dim]"
WIZARD_STEPS = ["Target", "Mode", "Execution", "Compute", "Parameters", "Flags"]


# ═══════════════════════════════════════════════════════════════════════
#  Data Model
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LaunchConfig:
    """Complete engagement launch configuration."""
    target_ip: str = ""
    mode: str = "ctf"              # normal | ctf | htb | stealth
    ctf_goal: str = "both"         # user | root | both
    execution: str = "live"        # simulated | live
    compute: str = "cloud"         # cpu | gpu | cloud | auto | remote
    steps: int = 500
    episodes: int = 1
    seed: Optional[int] = None
    dry_run: bool = False
    verbosity: str = "verbose"     # quiet | normal | verbose | debug
    feature_flags: Dict[str, str] = field(default_factory=dict)
    custom_env: Dict[str, str] = field(default_factory=dict)


@dataclass
class SystemInfo:
    """Detected environment capabilities."""
    gpu_available: bool = False
    gpu_name: str = "N/A"
    gpu_vram_gb: float = 0.0
    cuda_available: bool = False
    openai_key: bool = False
    local_llm_running: bool = False
    local_llm_model: str = "N/A"
    python_version: str = ""
    total_runs: int = 0
    last_run_name: str = ""


# ═══════════════════════════════════════════════════════════════════════
#  System Detection
# ═══════════════════════════════════════════════════════════════════════

def detect_system() -> SystemInfo:
    """Comprehensive environment detection."""
    info = SystemInfo(python_version=sys.version.split()[0])

    # ── GPU / CUDA ──────────────────────────────────────────────────
    try:
        import torch
        if torch.cuda.is_available():
            info.gpu_available = True
            info.gpu_name = torch.cuda.get_device_name(0)
            info.gpu_vram_gb = round(
                torch.cuda.get_device_properties(0).total_mem / (1024**3), 1
            )
            info.cuda_available = True
    except ImportError:
        pass

    # ── OpenAI API Key ──────────────────────────────────────────────
    info.openai_key = bool(os.environ.get("OPENAI_API_KEY"))

    # ── Local LLM Probe ────────────────────────────────────────────
    llm_port = int(os.environ.get("ARIASKA_LOCAL_LLM_PORT", "8192"))
    try:
        import urllib.request
        resp = urllib.request.urlopen(
            f"http://127.0.0.1:{llm_port}/v1/models", timeout=2
        )
        data = json.loads(resp.read())
        if data.get("data"):
            info.local_llm_running = True
            info.local_llm_model = data["data"][0].get("id", "unknown")
    except Exception:
        pass

    # ── Check for downloaded model files ───────────────────────────
    if not info.local_llm_running:
        for d in [_PROJECT_ROOT / "models", Path.home() / "models", Path("/models")]:
            if d.exists():
                gguf = list(d.glob("*.gguf"))
                if gguf:
                    info.local_llm_model = gguf[0].name
                    break

    # ── Training history ───────────────────────────────────────────
    artifacts = _PROJECT_ROOT / "artifacts"
    if artifacts.exists():
        runs = sorted(artifacts.glob("*_results.json"), reverse=True)
        info.total_runs = len(runs)
        if runs:
            info.last_run_name = runs[0].stem

    return info


# ═══════════════════════════════════════════════════════════════════════
#  System Status Panel
# ═══════════════════════════════════════════════════════════════════════

def build_system_panel(info: SystemInfo) -> Panel:
    """Rich system status panel — the first thing users see."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="cyan", width=16)
    table.add_column("Value", style="white")

    # Compute row — GPU + CUDA combined  
    if info.gpu_available:
        gpu_text = f"[{C_OK}]✓[/{C_OK}] {info.gpu_name} ({info.gpu_vram_gb}G VRAM)"
        if info.cuda_available:
            gpu_text += f"  [{C_OK}]CUDA ✓[/{C_OK}]"
    else:
        gpu_text = f"[{C_WARN}]✗[/{C_WARN}] CPU-only"
        if info.cuda_available:
            gpu_text += f"  [{C_OK}]CUDA ✓[/{C_OK}]"
    table.add_row("⚡ Compute", gpu_text)

    # LLM row — Local + Cloud combined
    llm_parts = []
    if info.local_llm_running:
        llm_parts.append(f"[{C_OK}]Local ✓[/{C_OK}] {info.local_llm_model}")
    elif info.local_llm_model != "N/A":
        llm_parts.append(f"[{C_DIM}]Local ✗ ({info.local_llm_model})[/{C_DIM}]")
    else:
        llm_parts.append(f"[{C_DIM}]Local ✗[/{C_DIM}]")

    if info.openai_key:
        llm_parts.append(f"[{C_OK}]API ✓[/{C_OK}]")
    else:
        llm_parts.append(f"[{C_ERR}]API ✗[/{C_ERR}]")
    table.add_row("🤖 LLM", "  │  ".join(llm_parts))

    # Runtime row — Python + History
    runtime_text = f"Python {info.python_version}"
    if info.total_runs > 0:
        runtime_text += f"  │  {info.total_runs} past run{'s' if info.total_runs != 1 else ''}"
    table.add_row("🐍 Runtime", runtime_text)

    return Panel(
        table,
        title="[bold]System Status[/bold]",
        border_style="bright_blue",
        box=box.ROUNDED,
        padding=(0, 2),
    )


# ═══════════════════════════════════════════════════════════════════════
#  Main Menu
# ═══════════════════════════════════════════════════════════════════════

def show_main_menu() -> str:
    """Full-featured command center menu with grouped sections."""
    console.print()

    menu_table = Table(
        show_header=False,
        box=box.SIMPLE_HEAVY,
        border_style=C_ACCENT,
        padding=(0, 2),
    )
    menu_table.add_column("Key", style=C_KEY, width=6, justify="center")
    menu_table.add_column("Action", style=C_LABEL)
    menu_table.add_column("Description", style=C_DESC)

    # ── ENGAGE ──────────────────────────────────────────────────────
    menu_table.add_row(
        "", f"[bold {C_ACCENT}]─── ENGAGE ───[/bold {C_ACCENT}]", ""
    )
    menu_table.add_row(
        "1", "🎯 Launch Wizard",
        "Full 6-step config: target → mode → compute → flags → launch"
    )
    menu_table.add_row(
        "2", "⚡ Quick Launch",
        "Enter IP and go — default settings, minimal prompts"
    )
    menu_table.add_row(
        "3", "🏴 CTF Mode",
        "Aggressive flag hunting — speed-first exploitation"
    )
    menu_table.add_row(
        "4", "🖥️  HTB Mode",
        "Realistic engagement — thorough recon + careful exploitation"
    )

    # ── OPERATE ─────────────────────────────────────────────────────
    menu_table.add_row(
        "", f"[bold {C_ACCENT}]─── OPERATE ───[/bold {C_ACCENT}]", ""
    )
    menu_table.add_row(
        "5", "📊 Training History",
        "Browse past runs, metrics, flag status, cost tracker"
    )
    menu_table.add_row(
        "6", "🔁 Resume / Replay",
        "Continue or replay from the most recent trace"
    )
    menu_table.add_row(
        "7", "🔧 GPU / LLM Manager",
        "Local LLM server, GPU health, model downloads"
    )

    # ── MAINTAIN ────────────────────────────────────────────────────
    menu_table.add_row(
        "", f"[bold {C_ACCENT}]─── MAINTAIN ───[/bold {C_ACCENT}]", ""
    )
    menu_table.add_row(
        "8", "🧪 Run Test Suite",
        "Quick / full / coverage / specific test modes"
    )
    menu_table.add_row(
        "9", "🔍 Diagnostics",
        "Health check — deps, core modules, disk, config"
    )
    menu_table.add_row(
        "0", "🛠️  Advanced Config",
        "Feature flags, PPO params, model routing, budget"
    )

    # ── EXIT ────────────────────────────────────────────────────────
    menu_table.add_row("", "", "")
    menu_table.add_row("q", f"[{C_DIM}]Exit[/{C_DIM}]", "")

    console.print(Panel(
        menu_table,
        title="[bold white] ═══ Command Center ═══ [/bold white]",
        border_style=C_ACCENT,
        box=box.HEAVY,
        padding=(1, 1),
    ))

    return Prompt.ask(
        f"[{C_KEY}]Select[/{C_KEY}]",
        choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "q"],
        default="1",
    )


# ═══════════════════════════════════════════════════════════════════════
#  UI Helpers
# ═══════════════════════════════════════════════════════════════════════

def _step_bar(current: int) -> str:
    """Render wizard step progress indicator."""
    parts: list[str] = []
    for i, name in enumerate(WIZARD_STEPS):
        if i < current:
            parts.append(f"{STEP_DONE} [green]{name}[/green]")
        elif i == current:
            parts.append(
                f"{STEP_ACTIVE} [bold {C_ACCENT}]{name}[/bold {C_ACCENT}]"
            )
        else:
            parts.append(f"{STEP_PENDING} [{C_DIM}]{name}[/{C_DIM}]")
    return "  ─  ".join(parts)


def _pick(
    title: str,
    prompt_text: str,
    options: list[tuple[str, str, str]],
    default: str = "1",
    border_style: str = C_DIM,
) -> str:
    """Elegant option picker with Panel display."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Key", style=C_KEY, width=4)
    table.add_column("Label", style=C_LABEL)
    table.add_column("Description", style=C_DESC)
    for key, label, desc in options:
        table.add_row(key, label, desc)
    console.print(Panel(table, title=title, border_style=border_style))

    keys = [o[0] for o in options]
    return Prompt.ask(prompt_text, choices=keys, default=default)


def _kv(key: str, val: str) -> str:
    """Format key: value pair."""
    return f"[{C_DIM}]{key}:[/{C_DIM}] [{C_LABEL}]{val}[/{C_LABEL}]"


# ═══════════════════════════════════════════════════════════════════════
#  Launch Wizard — 6-Step Flow
# ═══════════════════════════════════════════════════════════════════════

def wizard_launch(sys_info: SystemInfo) -> Optional[LaunchConfig]:
    """Full 6-step interactive launch wizard with visual progress."""
    cfg = LaunchConfig()
    console.print()
    console.print(Rule(
        f"[{C_TITLE}]🎯 Launch Engagement Wizard[/{C_TITLE}]",
        style=C_ACCENT,
    ))

    # ── Step 1: Target ──────────────────────────────────────────────
    console.print(f"\n  {_step_bar(0)}\n")
    cfg.target_ip = Prompt.ask(
        "🎯 Target IP or hostname",
        default="10.10.10.10",
    )
    if not cfg.target_ip or cfg.target_ip.count(".") < 1:
        console.print(f"[{C_WARN}]⚠ Invalid target — using 10.10.10.10[/{C_WARN}]")
        cfg.target_ip = "10.10.10.10"

    # ── Step 2: Mode ────────────────────────────────────────────────
    console.print(f"\n  {_step_bar(1)}\n")
    mode = _pick("Engagement Mode", "Mode", [
        ("1", "🔄 Normal",
         "Standard engagement — full kill chain, balanced exploration"),
        ("2", "🏴 CTF — User Flag",
         "Flag hunting — hunt user.txt only, speed-optimized"),
        ("3", "🏴 CTF — Root Flag",
         "Flag hunting — hunt root.txt only, privesc-focused"),
        ("4", "🏴 CTF — Both Flags",
         "Flag hunting — auto-close on user.txt + root.txt"),
        ("5", "🖥️  HTB Mode",
         "HackTheBox — realistic engagement, careful enumeration"),
        ("6", "👻 Stealth Mode",
         "Minimal detection risk — ShadowAgent priority, slow scanning"),
    ], default="4")
    mode_map = {
        "1": ("normal", "both"),
        "2": ("ctf", "user"),
        "3": ("ctf", "root"),
        "4": ("ctf", "both"),
        "5": ("htb", "both"),
        "6": ("stealth", "both"),
    }
    cfg.mode, cfg.ctf_goal = mode_map[mode]

    # ── Step 3: Execution ───────────────────────────────────────────
    console.print(f"\n  {_step_bar(2)}\n")
    exec_choice = _pick("Execution Mode", "Execution", [
        ("1", "🔴 Live (SSH)",
         "Execute real commands against the target via SSH/terminal"),
        ("2", "🔵 Simulated (Dry Run)",
         "No real commands — safe for testing, uses simulated output"),
    ], default="1")
    cfg.execution = "live" if exec_choice == "1" else "simulated"
    cfg.dry_run = cfg.execution == "simulated"

    # ── Step 4: Compute ─────────────────────────────────────────────
    console.print(f"\n  {_step_bar(3)}\n")
    compute_opts: list[tuple[str, str, str]] = []
    if sys_info.gpu_available and sys_info.openai_key:
        compute_opts.append(
            ("1", "🔀 Auto (Hybrid)",
             f"GPU for nano/mini, OpenAI for codex/full — "
             f"{sys_info.gpu_name} ({sys_info.gpu_vram_gb}G)")
        )
    if sys_info.openai_key:
        compute_opts.append(
            ("2", "☁️  Cloud (OpenAI Only)",
             "All LLM calls via OpenAI API (max quality, API cost)")
        )
    if sys_info.gpu_available:
        compute_opts.append(
            ("3", f"🖥️  GPU Only ({sys_info.gpu_name.split()[-1]})",
             f"All LLM calls via local GPU — {sys_info.gpu_vram_gb}G VRAM, no API cost")
        )
    compute_opts.append(
        ("4", "💻 CPU Only",
         "Offline / deterministic mode — no GPU, no API")
    )
    if sys_info.openai_key:
        compute_opts.append(
            ("5", "🌐 Remote GPU (vast.ai)",
             "SSH to cloud GPU for compute — requires vast.ai configured")
        )

    # Smart default: auto if GPU+key, cloud if key only, cpu otherwise
    if sys_info.gpu_available and sys_info.openai_key:
        default_compute = "1"
    elif sys_info.openai_key:
        default_compute = "2"
    elif sys_info.gpu_available:
        default_compute = "3"
    else:
        default_compute = "4"

    comp = _pick("Compute Backend", "Compute", compute_opts, default=default_compute)
    cfg.compute = {
        "1": "auto", "2": "cloud", "3": "gpu", "4": "cpu", "5": "remote",
    }[comp]

    # ── Step 5: Parameters ──────────────────────────────────────────
    console.print(f"\n  {_step_bar(4)}\n")
    cfg.steps = IntPrompt.ask(
        "📏 Max steps per episode", default=500
    )
    cfg.episodes = IntPrompt.ask(
        "🔁 Episode count", default=1
    )
    seed_str = Prompt.ask(
        "🎲 Random seed (empty = random)", default=""
    )
    cfg.seed = int(seed_str) if seed_str.strip().isdigit() else None

    cfg.verbosity = Prompt.ask(
        "📢 Verbosity",
        choices=["quiet", "normal", "verbose", "debug"],
        default="verbose",
    )

    # ── Step 6: Feature Flags ───────────────────────────────────────
    console.print(f"\n  {_step_bar(5)}\n")
    if Confirm.ask("🚩 Configure feature flags?", default=False):
        cfg.feature_flags = _flag_picker()
    else:
        console.print(f"[{C_DIM}]Using defaults — auto-detected profile.[/{C_DIM}]")

    # ── Confirmation ────────────────────────────────────────────────
    console.print()
    _show_config_summary(cfg)
    console.print()

    if not Confirm.ask(
        "[bold green]🚀 Launch engagement?[/bold green]", default=True
    ):
        console.print(f"[{C_WARN}]Cancelled — returning to menu.[/{C_WARN}]")
        return None

    return cfg


def _flag_picker() -> Dict[str, str]:
    """Interactive feature flag override picker with grouped display."""
    flag_groups: dict[str, list[tuple[str, str, str]]] = {
        "RL & Training": [
            ("FF_BC_LOSS", "ON", "Behavioral Cloning loss from TeacherTrace"),
            ("FF_TEACHER_TRACE", "ON", "Teacher distillation pipeline"),
            ("FF_NEUROMODULATORS", "ON", "Biologically-inspired RL control signals"),
        ],
        "Intelligence Pipeline": [
            ("FF_USE_MICRO_CHAIN", "ON", "MicroChain 3-stage nano→mini→nano scoring"),
            ("FF_EVIDENCE_GRAPH", "ON", "Evidence graph reasoning"),
            ("FF_HYPOTHESIS_ENGINE", "ON", "Hypothesis generation + validation"),
            ("FF_STRICT_EXPLOIT_GATE", "enforce", "Evidence gate: off | log | enforce"),
        ],
        "Ops & Safety (Phase 38)": [
            ("FF_OPS_HUB", "ON", "OpsHub pre-flight + post-step orchestration"),
            ("FF_OPS_PREFLIGHT", "ON", "OpsHub pre-flight authority"),
            ("FF_PHASE_INVARIANTS", "ON", "Phase hardening + shell validation"),
            ("FF_COMMAND_LOCKOUT", "ON", "Anti-repeat + cooldown enforcement"),
            ("FF_SHELL_VALIDATOR", "ON", "Shell session validation"),
            ("FF_EXPLOIT_COOLDOWN", "ON", "Exploit cooldown management"),
            ("FF_EXPLOIT_CONFIDENCE", "ON", "Exploit confidence scoring"),
            ("FF_DISCOVERY_TRUST", "ON", "Discovery trust engine"),
        ],
        "Phase 39 Features": [
            ("FF_ORION_RETHINK", "ON", "Orion deep-rethink escalation"),
            ("FF_TRUST_WEIGHTS", "ON", "Trust weight annealing per agent"),
            ("FF_CAP_GATE", "ON", "CAP regression gate"),
            ("FF_DEBUG_TRACE", "ON", "Debug trace instrumentation"),
        ],
        "Multi-Agent": [
            ("FF_PARALLEL_AGENTS", "ON", "Multi-agent parallel activation"),
            ("FF_STRICT_PHASE_LADDER", "ON", "Phase ordering enforcement"),
        ],
    }

    for group_name, flags in flag_groups.items():
        table = Table(show_header=True, box=box.SIMPLE, padding=(0, 1))
        table.add_column("Flag", style="cyan", width=30)
        table.add_column("Default", style=C_OK, width=8)
        table.add_column("Description", style=C_DIM, width=46)
        for env_key, default, desc in flags:
            def_style = C_OK if default in ("ON", "enforce") else C_ERR
            table.add_row(env_key, f"[{def_style}]{default}[/{def_style}]", desc)
        console.print(Panel(
            table,
            title=f"[bold]{group_name}[/bold]",
            border_style=C_DIM,
            box=box.ROUNDED,
        ))

    console.print()
    toggle_str = Prompt.ask(
        "🚩 Override flags (comma-separated, e.g. "
        "[cyan]FF_BC_LOSS=0,FF_STRICT_EXPLOIT_GATE=log[/cyan])\n"
        f"   [{C_DIM}]Press Enter to keep all defaults[/{C_DIM}]",
        default="",
    )

    overrides: Dict[str, str] = {}
    if toggle_str.strip():
        for pair in toggle_str.split(","):
            pair = pair.strip()
            if "=" in pair:
                k, v = pair.split("=", 1)
                overrides[k.strip()] = v.strip()

    return overrides


def _show_config_summary(cfg: LaunchConfig) -> None:
    """Polished configuration summary panel."""
    table = Table(
        show_header=False, box=None, padding=(0, 2),
    )
    table.add_column("Parameter", style="cyan", width=14)
    table.add_column("Value", style="white")

    # Target
    table.add_row("🎯 Target", f"[bold white]{cfg.target_ip}[/bold white]")

    # Mode
    mode_display = {
        "normal": "🔄 Normal — Full kill chain",
        "htb": "🖥️  HTB — Realistic engagement",
        "stealth": "👻 Stealth — Low detection risk",
    }.get(cfg.mode, cfg.mode)
    if cfg.mode == "ctf":
        goal_display = {"user": "User Flag 🏴", "root": "Root Flag 🏴", "both": "Both Flags 🏴🏴"}
        mode_display = f"🏴 CTF → {goal_display[cfg.ctf_goal]}"
    table.add_row("📋 Mode", mode_display)

    # Execution
    exec_display = (
        "[bold red]🔴 LIVE[/bold red] — Real commands via SSH"
        if cfg.execution == "live"
        else "[bold blue]🔵 SIMULATED[/bold blue] — Dry run, no real execution"
    )
    table.add_row("⚙️  Execution", exec_display)

    # Compute
    compute_display = {
        "auto": "🔀 Auto (GPU + API hybrid)",
        "cloud": "☁️  Cloud (OpenAI API)",
        "gpu": "🖥️  GPU Only (local)",
        "cpu": "💻 CPU Only (offline)",
        "remote": "🌐 Remote GPU (vast.ai)",
    }.get(cfg.compute, cfg.compute)
    table.add_row("🖥️  Compute", compute_display)

    # Steps
    ep_label = f"episode{'s' if cfg.episodes > 1 else ''}"
    table.add_row("📏 Steps", f"{cfg.steps} × {cfg.episodes} {ep_label}")

    # Seed
    table.add_row("🎲 Seed", str(cfg.seed) if cfg.seed else "random")

    # Verbosity
    table.add_row("📢 Verbosity", cfg.verbosity)

    # Feature flag overrides
    if cfg.feature_flags:
        flag_str = ", ".join(f"{k}={v}" for k, v in cfg.feature_flags.items())
        table.add_row("🚩 Flags", flag_str)

    console.print(Panel(
        table,
        title=f"[{C_TITLE}] ═══ Configuration Summary ═══ [{C_TITLE}]",
        border_style=C_OK,
        box=box.HEAVY,
        padding=(1, 2),
    ))


# ═══════════════════════════════════════════════════════════════════════
#  Quick Launch / CTF / HTB — Minimal-Prompt Modes
# ═══════════════════════════════════════════════════════════════════════

def quick_launch(sys_info: SystemInfo) -> Optional[LaunchConfig]:
    """Minimal-prompt launch — enter IP and go."""
    console.print()
    console.print(Rule(
        f"[{C_TITLE}]⚡ Quick Launch[/{C_TITLE}]", style=C_ACCENT
    ))
    console.print(
        f"[{C_DIM}]Default settings, minimal prompts. "
        f"For full control use option 1.[/{C_DIM}]\n"
    )

    target = Prompt.ask("🎯 Target IP", default="10.10.10.10")
    steps = IntPrompt.ask("📏 Steps", default=200)

    cfg = LaunchConfig(
        target_ip=target,
        mode="normal",
        execution="live",
        compute=_auto_compute(sys_info),
        steps=steps,
        verbosity="verbose",
    )

    _show_config_summary(cfg)
    console.print()
    if not Confirm.ask("[bold green]🚀 Launch?[/bold green]", default=True):
        return None
    return cfg


def quick_ctf(sys_info: SystemInfo) -> Optional[LaunchConfig]:
    """CTF-optimized launch — flag hunting mode."""
    console.print()
    console.print(Rule(
        f"[{C_TITLE}]🏴 CTF Mode — Flag Hunter[/{C_TITLE}]",
        style="bright_red",
    ))
    console.print(
        f"[{C_DIM}]Aggressive exploitation, speed-optimized. "
        f"Targets user.txt + root.txt.[/{C_DIM}]\n"
    )

    target = Prompt.ask("🎯 Target IP", default="10.10.10.10")
    steps = IntPrompt.ask("📏 Max steps", default=500)
    seed_str = Prompt.ask("🎲 Seed (empty = random)", default="")
    seed = int(seed_str) if seed_str.strip().isdigit() else None

    cfg = LaunchConfig(
        target_ip=target,
        mode="ctf",
        ctf_goal="both",
        execution="live",
        compute=_auto_compute(sys_info),
        steps=steps,
        seed=seed,
        verbosity="verbose",
    )

    _show_config_summary(cfg)
    console.print()
    if not Confirm.ask("[bold green]🚀 Launch?[/bold green]", default=True):
        return None
    return cfg


def quick_htb(sys_info: SystemInfo) -> Optional[LaunchConfig]:
    """HTB-optimized launch — realistic engagement profile."""
    console.print()
    console.print(Rule(
        f"[{C_TITLE}]🖥️  HTB Mode — Realistic Engagement[/{C_TITLE}]",
        style="bright_green",
    ))
    console.print(
        f"[{C_DIM}]Thorough enumeration, careful exploitation. "
        f"Full kill chain with flag detection.[/{C_DIM}]\n"
    )

    target = Prompt.ask("🎯 Target IP", default="10.10.10.10")
    steps = IntPrompt.ask("📏 Max steps", default=300)

    cfg = LaunchConfig(
        target_ip=target,
        mode="htb",
        ctf_goal="both",
        execution="live",
        compute=_auto_compute(sys_info),
        steps=steps,
        verbosity="verbose",
    )

    _show_config_summary(cfg)
    console.print()
    if not Confirm.ask("[bold green]🚀 Launch?[/bold green]", default=True):
        return None
    return cfg


def _auto_compute(sys_info: SystemInfo) -> str:
    """Determine best compute backend automatically."""
    if sys_info.gpu_available and sys_info.openai_key:
        return "auto"
    if sys_info.openai_key:
        return "cloud"
    if sys_info.gpu_available:
        return "gpu"
    return "cpu"


# ═══════════════════════════════════════════════════════════════════════
#  Resume Last Training
# ═══════════════════════════════════════════════════════════════════════

def resume_last_training() -> None:
    """Resume or replay the most recent training trace."""
    console.print()
    console.print(Rule(
        f"[{C_TITLE}]🔁 Resume Last Training[/{C_TITLE}]",
        style=C_ACCENT,
    ))

    traces_dir = _PROJECT_ROOT / "traces"
    if not traces_dir.exists():
        console.print(f"[{C_WARN}]No traces directory found.[/{C_WARN}]")
        return

    trace_files = sorted(traces_dir.glob("*.jsonl"), reverse=True)
    if not trace_files:
        console.print(f"[{C_WARN}]No trace files found.[/{C_WARN}]")
        return

    table = Table(
        box=box.SIMPLE, border_style=C_DIM, padding=(0, 1),
        title=f"[bold]Recent Traces ({len(trace_files)} total)[/bold]",
    )
    table.add_column("#", style=C_KEY, width=4)
    table.add_column("Trace File", style="cyan")
    table.add_column("Size", style=C_DIM, width=10, justify="right")

    for i, f in enumerate(trace_files[:10], 1):
        size_kb = f.stat().st_size / 1024
        table.add_row(str(i), f.name, f"{size_kb:.1f} KB")

    console.print(table)
    console.print()

    idx = IntPrompt.ask("Which # to replay?", default=1)
    if 1 <= idx <= min(10, len(trace_files)):
        trace = trace_files[idx - 1]
        console.print(f"\n[{C_ACCENT}]Replaying: {trace.name}[/{C_ACCENT}]")
        cmd = [
            sys.executable, "ariaska_cli.py", "replay",
            str(trace), "--verbose",
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_PROJECT_ROOT)
        subprocess.run(cmd, cwd=str(_PROJECT_ROOT), env=env)


# ═══════════════════════════════════════════════════════════════════════
#  Execution Engine
# ═══════════════════════════════════════════════════════════════════════

def build_command(cfg: LaunchConfig) -> List[str]:
    """Assemble the CLI command from launch config."""
    cmd = [
        sys.executable, "ariaska_cli.py", "smart-train",
        "--target", cfg.target_ip,
        "--steps", str(cfg.steps),
    ]
    if cfg.mode in ("ctf", "htb"):
        cmd.append("--ctf")
    if cfg.seed is not None:
        cmd.extend(["--seed", str(cfg.seed)])
    return cmd


def build_env(cfg: LaunchConfig) -> Dict[str, str]:
    """Build environment variables for the child process."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_PROJECT_ROOT)

    if cfg.dry_run:
        env["ARIASKA_DRY_RUN"] = "1"

    # Compute mode
    if cfg.compute == "gpu":
        env["FF_LOCAL_LLM"] = "1"
        env["FF_LOCAL_LLM_OFFLOAD_NANO"] = "1"
        env["FF_LOCAL_LLM_OFFLOAD_MINI"] = "1"
    elif cfg.compute == "cpu":
        env["FF_LOCAL_LLM"] = "0"
    elif cfg.compute == "auto":
        env["FF_LOCAL_LLM"] = "1"

    # Mode-specific
    if cfg.mode == "stealth":
        env.setdefault("FF_SHADOW_PRIORITY", "1")

    # Feature flag overrides
    for k, v in cfg.feature_flags.items():
        env[k] = v

    # Custom env
    for k, v in cfg.custom_env.items():
        env[k] = v

    return env


def execute(cfg: LaunchConfig) -> None:
    """Launch the engagement and handle post-run bookkeeping."""
    cmd = build_command(cfg)
    env = build_env(cfg)

    # ── Prepare run directory ───────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_prefix = cfg.mode if cfg.mode != "normal" else "run"
    run_name = f"{mode_prefix}_{timestamp}_{cfg.target_ip}"
    run_dir = _PROJECT_ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "live_output.log"

    # Persist config
    config_path = run_dir / "config.json"
    import dataclasses
    with open(config_path, "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2, default=str)

    # ── Launch panel ────────────────────────────────────────────────
    console.print()
    console.print(Rule(
        f"[bold green]🚀 Launching Ariaska Engagement[/bold green]",
        style=C_OK,
    ))

    launch_info = Table(show_header=False, box=None, padding=(0, 2))
    launch_info.add_column("Key", style="cyan", width=12)
    launch_info.add_column("Value", style="white")
    launch_info.add_row("Command", f"[{C_DIM}]{' '.join(cmd)}[/{C_DIM}]")
    launch_info.add_row("Log", f"[{C_DIM}]{log_path}[/{C_DIM}]")
    launch_info.add_row("Config", f"[{C_DIM}]{config_path}[/{C_DIM}]")
    launch_info.add_row("Run Dir", f"[{C_DIM}]{run_dir}[/{C_DIM}]")

    console.print(Panel(
        launch_info,
        title=f"[bold {C_ACCENT}] ▶ Engagement Starting [/bold {C_ACCENT}]",
        border_style=C_ACCENT,
        box=box.HEAVY,
        padding=(1, 2),
    ))
    console.print()

    if cfg.compute == "remote":
        console.print(
            f"[{C_WARN}]⚠ Remote GPU mode — ensure vast.ai SSH "
            f"is configured before proceeding.[/{C_WARN}]\n"
        )

    t0 = time.time()

    try:
        os.chdir(str(_PROJECT_ROOT))
        full_cmd = " ".join(cmd) + f" 2>&1 | tee {log_path}"
        os.execve("/bin/sh", ["/bin/sh", "-c", full_cmd], env)
    except Exception as e:
        console.print(
            f"[{C_ERR}]execve failed: {e} — falling back to subprocess[/{C_ERR}]"
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
            console.print(f"\n[{C_WARN}]Interrupted by user.[/{C_WARN}]")

    elapsed = time.time() - t0

    # ── Post-run artifact sync ──────────────────────────────────────
    console.print()
    console.print(Rule(
        "[bold]📦 Post-Run Artifact Sync[/bold]", style=C_OK
    ))
    _sync_artifacts(run_dir, elapsed)


def _sync_artifacts(run_dir: Path, elapsed: float) -> None:
    """Copy results and traces into the run directory with summary."""
    import shutil

    artifacts_dir = _PROJECT_ROOT / "artifacts"
    results = (
        sorted(artifacts_dir.glob("*_results.json"), reverse=True)
        if artifacts_dir.exists() else []
    )

    if results:
        latest = results[0]
        try:
            shutil.copy2(latest, run_dir / "results.json")
            data = json.loads(latest.read_text())

            uf = data.get("user_flag_captured", False)
            rf = data.get("root_flag_captured", False)
            steps = data.get("total_steps",
                             data.get("episode_metrics", {}).get("total_steps", "?"))
            reward = data.get("total_reward", 0)
            phase = data.get("highest_phase",
                             data.get("episode_metrics", {}).get("highest_phase", "?"))
            cost = data.get("total_cost_usd", data.get("cost_usd", 0))

            status = (
                f"[bold {C_OK}]🏆 PWNED[/bold {C_OK}]"
                if uf and rf
                else f"[{C_WARN}]INCOMPLETE[/{C_WARN}]"
            )

            summary = Table(show_header=False, box=None, padding=(0, 2))
            summary.add_column("Key", style="cyan", width=14)
            summary.add_column("Value", style="white")
            summary.add_row("Result", status)
            summary.add_row("Steps", str(steps))
            summary.add_row("Reward", f"{reward:+.1f}" if isinstance(reward, (int, float)) else str(reward))
            summary.add_row("Highest Phase", str(phase))
            summary.add_row("Duration", f"{elapsed:.0f}s ({elapsed/60:.1f}m)")
            if isinstance(cost, (int, float)) and cost > 0:
                summary.add_row("API Cost", f"${cost:.3f}")
            if uf:
                val = data.get("user_flag_value", "captured")
                summary.add_row("🏴 User Flag", f"[{C_OK}]{val}[/{C_OK}]")
            if rf:
                val = data.get("root_flag_value", "captured")
                summary.add_row("🏴 Root Flag", f"[{C_OK}]{val}[/{C_OK}]")

            console.print(Panel(
                summary,
                title=f"[{C_TITLE}] ═══ Engagement Results ═══ [{C_TITLE}]",
                border_style=C_OK if (uf and rf) else C_WARN,
                box=box.HEAVY,
                padding=(1, 2),
            ))
        except Exception:
            pass

    # Copy latest trace
    traces_dir = _PROJECT_ROOT / "traces"
    if traces_dir.exists():
        traces = sorted(traces_dir.glob("events_*.jsonl"), reverse=True)
        if traces:
            shutil.copy2(traces[0], run_dir / "trace.jsonl")
            console.print(
                f"\n📋 Trace:     [{C_DIM}]{traces[0].name} → {run_dir.name}/[/{C_DIM}]"
            )

    console.print(f"📁 Artifacts: [{C_DIM}]{run_dir}[/{C_DIM}]")
    console.print()


# ═══════════════════════════════════════════════════════════════════════
#  GPU / LLM Manager
# ═══════════════════════════════════════════════════════════════════════

def show_gpu_manager(sys_info: SystemInfo) -> None:
    """GPU and local LLM management submenu."""
    while True:
        console.print()
        console.print(Rule(
            f"[{C_TITLE}]🔧 GPU / LLM Manager[/{C_TITLE}]",
            style="bright_magenta",
        ))

        # Current status
        console.print(build_system_panel(sys_info))

        choice = _pick("Actions", "Select", [
            ("1", "▶️  Start Local LLM Server", "Launch llama-server with configured model"),
            ("2", "⏹️  Stop Local LLM Server", "Gracefully stop the running server"),
            ("3", "🩺 Health Check", "Test connectivity + inference latency"),
            ("4", "📊 GPU Utilization", "Run nvidia-smi for VRAM/utilization stats"),
            ("5", "📥 Download Model", "Download GGUF model from HuggingFace"),
            ("6", "📋 View LLM Logs", "Tail the last 50 lines of LLM server logs"),
            ("b", "↩️  Back to Main Menu", ""),
        ], default="b", border_style="bright_magenta")

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
        elif choice == "6":
            _view_llm_logs()

        # Refresh status
        refreshed = detect_system()
        sys_info.local_llm_running = refreshed.local_llm_running
        sys_info.local_llm_model = refreshed.local_llm_model


def _start_llm() -> None:
    """Start local LLM server."""
    try:
        from core.llm.local_llm_provider import get_local_llm_provider
        provider = get_local_llm_provider()
        if provider.is_available():
            console.print(f"[{C_WARN}]LLM server already running.[/{C_WARN}]")
            return
        console.print(f"[{C_ACCENT}]Starting local LLM server...[/{C_ACCENT}]")
        if provider.start_server():
            console.print(f"[{C_OK}]✓ LLM server started successfully.[/{C_OK}]")
        else:
            console.print(f"[{C_ERR}]✗ Failed to start LLM server.[/{C_ERR}]")
    except Exception as e:
        console.print(f"[{C_ERR}]Error: {e}[/{C_ERR}]")


def _stop_llm() -> None:
    """Stop local LLM server."""
    try:
        from core.llm.local_llm_provider import get_local_llm_provider
        get_local_llm_provider().stop_server()
        console.print(f"[{C_OK}]✓ LLM server stopped.[/{C_OK}]")
    except Exception as e:
        console.print(f"[{C_ERR}]Error: {e}[/{C_ERR}]")


def _check_llm() -> None:
    """Check LLM server health and run inference test."""
    port = int(os.environ.get("ARIASKA_LOCAL_LLM_PORT", "8192"))
    try:
        import urllib.request

        resp = urllib.request.urlopen(
            f"http://127.0.0.1:{port}/v1/models", timeout=5
        )
        models = json.loads(resp.read())
        console.print(f"[{C_OK}]✓ Server responding on port {port}[/{C_OK}]")
        if models.get("data"):
            for m in models["data"]:
                console.print(f"  Model: [bold]{m.get('id', 'unknown')}[/bold]")

        # Inference test
        console.print(f"[{C_ACCENT}]Running inference test...[/{C_ACCENT}]")
        req_data = json.dumps({
            "model": "local",
            "messages": [
                {"role": "user", "content": "Say 'Ariaska ready' in exactly 2 words"}
            ],
            "max_tokens": 10,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=req_data,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        resp = urllib.request.urlopen(req, timeout=30)
        latency_ms = (time.time() - t0) * 1000
        result = json.loads(resp.read())
        content = result["choices"][0]["message"]["content"].strip()
        tokens = result.get("usage", {}).get("total_tokens", "?")
        console.print(f"  Response: [bold]{content}[/bold]")
        console.print(f"  Latency:  {latency_ms:.0f}ms | Tokens: {tokens}")
        console.print(f"[{C_OK}]✓ Inference test passed.[/{C_OK}]")
    except Exception as e:
        console.print(f"[{C_ERR}]✗ Health check failed: {e}[/{C_ERR}]")


def _nvidia_smi() -> None:
    """Show nvidia-smi output."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            console.print(Panel(
                result.stdout,
                title="nvidia-smi",
                border_style=C_OK,
                box=box.ROUNDED,
            ))
        else:
            console.print(f"[{C_WARN}]nvidia-smi not available.[/{C_WARN}]")
    except FileNotFoundError:
        console.print(
            f"[{C_WARN}]nvidia-smi not found — no NVIDIA GPU detected.[/{C_WARN}]"
        )


def _download_model() -> None:
    """Download a GGUF model from HuggingFace."""
    console.print()
    console.print(Rule("📥 Model Download", style=C_DIM))

    table = Table(show_header=True, box=box.SIMPLE, padding=(0, 1))
    table.add_column("#", style=C_KEY, width=3)
    table.add_column("Model", style="white")
    table.add_column("Size", style=C_DIM)
    table.add_column("VRAM Required", style=C_DIM)
    table.add_row("1", "Qwen3-32B-Instruct Q4_K_M", "~18.5 GB", "~20 GB")
    table.add_row("2", "Qwen3-32B-Instruct Q5_K_M", "~22 GB", "~24 GB")
    table.add_row("3", "Qwen3-14B-Instruct Q5_K_M", "~10 GB", "~12 GB")
    table.add_row("4", "Qwen3-8B-Instruct Q8_0", "~8 GB", "~10 GB")
    console.print(table)

    choice = Prompt.ask("Select model", choices=["1", "2", "3", "4"], default="1")

    model_info = {
        "1": ("Qwen/Qwen3-32B-Instruct-GGUF", "Qwen3-32B-Instruct-Q4_K_M.gguf"),
        "2": ("Qwen/Qwen3-32B-Instruct-GGUF", "Qwen3-32B-Instruct-Q5_K_M.gguf"),
        "3": ("Qwen/Qwen3-14B-Instruct-GGUF", "Qwen3-14B-Instruct-Q5_K_M.gguf"),
        "4": ("Qwen/Qwen3-8B-Instruct-GGUF", "Qwen3-8B-Instruct-Q8_0.gguf"),
    }

    repo, filename = model_info[choice]
    model_dir = Path(
        os.environ.get("ARIASKA_MODEL_DIR", str(_PROJECT_ROOT / "models"))
    )
    model_dir.mkdir(parents=True, exist_ok=True)

    target = model_dir / filename
    if target.exists():
        console.print(f"[{C_WARN}]Model already exists: {target}[/{C_WARN}]")
        if not Confirm.ask("Re-download?", default=False):
            return

    console.print(f"[{C_ACCENT}]Downloading {filename}...[/{C_ACCENT}]")
    console.print(f"[{C_DIM}]This may take 10-30 minutes depending on connection speed.[/{C_DIM}]")

    subprocess.run([
        sys.executable, "-m", "huggingface_hub", "download",
        repo, filename,
        "--local-dir", str(model_dir),
        "--local-dir-use-symlinks", "False",
    ])


def _view_llm_logs() -> None:
    """View LLM server logs."""
    log_path = Path("/var/log/ariaska_llm.log")
    if not log_path.exists():
        console.print(
            f"[{C_WARN}]No log file found at /var/log/ariaska_llm.log[/{C_WARN}]"
        )
        return
    try:
        result = subprocess.run(
            ["tail", "-50", str(log_path)], capture_output=True, text=True
        )
        content = result.stdout[-3000:] if result.stdout else "Empty"
        console.print(Panel(
            content,
            title="LLM Server Logs (last 50 lines)",
            border_style=C_DIM,
            box=box.ROUNDED,
        ))
    except Exception as e:
        console.print(f"[{C_ERR}]Error reading logs: {e}[/{C_ERR}]")


# ═══════════════════════════════════════════════════════════════════════
#  Training History
# ═══════════════════════════════════════════════════════════════════════

def show_training_history() -> None:
    """Browse past training runs with metrics, flags, and cost tracking."""
    console.print()
    console.print(Rule(
        f"[{C_TITLE}]📊 Training History[/{C_TITLE}]",
        style="bright_yellow",
    ))

    artifacts_dir = _PROJECT_ROOT / "artifacts"
    result_files = (
        sorted(artifacts_dir.glob("*_results.json"), reverse=True)
        if artifacts_dir.exists() else []
    )

    if not result_files:
        console.print(f"[{C_DIM}]No training results found in artifacts/[/{C_DIM}]")
        return

    # ── Results table ───────────────────────────────────────────────
    table = Table(
        title=f"[bold]Recent Engagements ({len(result_files)} total)[/bold]",
        box=box.ROUNDED,
        border_style="yellow",
        padding=(0, 1),
    )
    table.add_column("#", style=C_DIM, width=4)
    table.add_column("Timestamp", style="white", width=16)
    table.add_column("Target", style="cyan", width=16)
    table.add_column("Steps", style="white", width=7, justify="right")
    table.add_column("Reward", style="yellow", width=9, justify="right")
    table.add_column("Phase", style="bright_magenta", width=18)
    table.add_column("Discoveries", style="cyan", width=12, justify="right")
    table.add_column("Flags", style=C_OK, width=14)
    table.add_column("Cost", style=C_ERR, width=8, justify="right")

    for i, f in enumerate(result_files[:20], 1):
        try:
            data = json.loads(f.read_text())
            ep = data.get("episode_metrics", {})

            # Timestamp
            ts = data.get("timestamp", "")[:16]
            if not ts and "engagement_" in f.stem:
                parts = f.stem.split("_")
                if len(parts) >= 2:
                    ts = parts[1][:8]

            # Metrics
            target = data.get("target_ip", "?")
            steps_val = data.get("total_steps", ep.get("total_steps", "?"))
            reward = data.get("total_reward", 0)
            reward_str = f"{reward:+.1f}" if isinstance(reward, (int, float)) else str(reward)
            phase = data.get("highest_phase", ep.get("highest_phase", "?"))
            disc = data.get("total_discoveries", ep.get("total_discoveries", "?"))
            cost = data.get("total_cost_usd", data.get("cost_usd", 0))
            cost_str = f"${cost:.3f}" if isinstance(cost, (int, float)) and cost > 0 else "—"

            # Flags
            uf = data.get("user_flag_captured", False)
            rf = data.get("root_flag_captured", False)
            if uf and rf:
                flags_str = f"[bold {C_OK}]🏴🏴 PWNED[/bold {C_OK}]"
            elif uf:
                flags_str = f"[{C_OK}]🏴 user[/{C_OK}]"
            elif rf:
                flags_str = f"[{C_OK}]🏴 root[/{C_OK}]"
            else:
                flags_str = f"[{C_DIM}]—[/{C_DIM}]"

            table.add_row(
                str(i), ts, target, str(steps_val),
                reward_str, phase, str(disc), flags_str, cost_str,
            )
        except Exception:
            table.add_row(str(i), f.name[:16], *["?"] * 6, "?")

    console.print(table)

    # ── Run directories ─────────────────────────────────────────────
    runs_dir = _PROJECT_ROOT / "runs"
    if runs_dir.exists():
        run_dirs = sorted(
            [d for d in runs_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name, reverse=True,
        )
        if run_dirs:
            console.print(
                f"\n[{C_DIM}]📁 {len(run_dirs)} run directories in runs/[/{C_DIM}]"
            )
            for d in run_dirs[:8]:
                n = len(list(d.iterdir()))
                console.print(
                    f"   [{C_DIM}]📁[/{C_DIM}] {d.name}  [{C_DIM}]({n} files)[/{C_DIM}]"
                )

    # ── Cost rollup ─────────────────────────────────────────────────
    total_cost = 0.0
    for f in result_files:
        try:
            data = json.loads(f.read_text())
            total_cost += float(data.get("total_cost_usd", data.get("cost_usd", 0)))
        except Exception:
            pass
    if total_cost > 0:
        console.print(
            f"\n[bold]Total API cost across all runs: "
            f"[{C_ERR}]${total_cost:.2f}[/{C_ERR}][/bold]"
        )

    # ── Detail view ─────────────────────────────────────────────────
    console.print()
    if result_files and Confirm.ask("View detailed results?", default=False):
        idx = IntPrompt.ask("Which # to view?", default=1)
        if 1 <= idx <= len(result_files):
            data = json.loads(result_files[idx - 1].read_text())
            console.print_json(json.dumps(data, indent=2, default=str)[:6000])

    console.print()


# ═══════════════════════════════════════════════════════════════════════
#  Test Suite Runner
# ═══════════════════════════════════════════════════════════════════════

def run_tests() -> None:
    """Run the project test suite."""
    console.print()
    console.print(Rule(
        f"[{C_TITLE}]🧪 Test Suite[/{C_TITLE}]",
        style="bright_green",
    ))

    mode = _pick("Test Mode", "Mode", [
        ("1", "⚡ Quick", "Stop on first failure, short tracebacks (-x --tb=short)"),
        ("2", "📋 Full", "Run all tests, don't stop on failures"),
        ("3", "📊 Coverage", "Full run + coverage report for core/"),
        ("4", "🔎 Specific", "Specify a test file or grep pattern"),
    ], default="1", border_style="bright_green")

    base_env = dict(os.environ)
    base_env["ARIASKA_DRY_RUN"] = "1"
    base_env["PYTHONPATH"] = str(_PROJECT_ROOT)
    pytest_bin = str(_PROJECT_ROOT / ".venv/bin/pytest")
    ignore = "--ignore=tests/test_online_makes_mentor_calls.py"

    if mode == "1":
        cmd = [pytest_bin, "tests/", "-x", "--tb=short", "-q",
               "--timeout=120", ignore]
    elif mode == "2":
        cmd = [pytest_bin, "tests/", "--tb=short", "-q",
               "--timeout=120", ignore]
    elif mode == "3":
        cmd = [pytest_bin, "tests/", "--tb=short", "-q", "--timeout=120",
               "--cov=core", "--cov-report=term-missing", ignore]
    else:
        path = Prompt.ask("Test file or pattern", default="tests/")
        cmd = [pytest_bin, path, "-v", "--tb=short", "--timeout=120"]

    console.print(f"\n[{C_DIM}]Running: {' '.join(cmd)}[/{C_DIM}]\n")
    subprocess.run(cmd, env=base_env, cwd=str(_PROJECT_ROOT))
    console.print()


# ═══════════════════════════════════════════════════════════════════════
#  System Diagnostics
# ═══════════════════════════════════════════════════════════════════════

def show_diagnostics() -> None:
    """Comprehensive system health check."""
    console.print()
    console.print(Rule(
        f"[{C_TITLE}]🔍 System Diagnostics[/{C_TITLE}]",
        style="bright_red",
    ))

    # System status
    sys_info = detect_system()
    console.print(build_system_panel(sys_info))

    # ── Dependency Health ───────────────────────────────────────────
    console.print("\n[bold]📦 Dependency Health Check[/bold]")
    deps = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("rich", "Rich"),
        ("openai", "OpenAI SDK"),
        ("llama_cpp", "llama-cpp-python"),
        ("chromadb", "ChromaDB"),
        ("sentence_transformers", "SentenceTransformers"),
    ]
    for module, name in deps:
        try:
            m = __import__(module)
            ver = getattr(m, "__version__", "?")
            console.print(f"  [{C_OK}]✓[/{C_OK}] {name:<24} {ver}")
        except ImportError:
            console.print(
                f"  [{C_ERR}]✗[/{C_ERR}] {name:<24} [{C_DIM}]not installed[/{C_DIM}]"
            )

    # ── Core Module Health ──────────────────────────────────────────
    console.print("\n[bold]🧠 Core Module Health[/bold]")
    core_modules = [
        "core.gpt_manager",
        "core.feature_flags",
        "core.runtime_flags",
        "core.llm.budget_manager",
        "core.llm.smart_mentor",
        "core.llm.micro_chain",
        "core.llm.phase_guided_llm",
        "core.llm.llm_policy_bridge",
        "core.algorithms.ppo_agent",
        "core.models.state_encoder",
        "core.training.smart_coach",
        "core.orchestration.smart_orchestrator",
    ]
    for mod in core_modules:
        try:
            __import__(mod)
            console.print(f"  [{C_OK}]✓[/{C_OK}] {mod}")
        except Exception as e:
            err_msg = str(e)[:60]
            console.print(f"  [{C_ERR}]✗[/{C_ERR}] {mod}  [{C_DIM}]{err_msg}[/{C_DIM}]")

    # ── Disk Usage ──────────────────────────────────────────────────
    console.print("\n[bold]💾 Disk Usage[/bold]")
    for d in [
        "data/", "models/", "artifacts/", "traces/", "runs/",
        "postmortems/", "logs/", "cache/",
    ]:
        p = _PROJECT_ROOT / d
        if p.exists():
            try:
                result = subprocess.run(
                    ["du", "-sh", str(p)],
                    capture_output=True, text=True, timeout=10,
                )
                size = result.stdout.split()[0] if result.stdout else "?"
                console.print(f"  {d:<20} {size}")
            except Exception:
                console.print(f"  {d:<20} ?")

    # ── Configuration ───────────────────────────────────────────────
    console.print("\n[bold]⚙️  Configuration[/bold]")
    env_ok = (_PROJECT_ROOT / ".env").exists()
    env_status = (
        f"[{C_OK}]✓ Found[/{C_OK}]"
        if env_ok else f"[{C_ERR}]✗ Missing[/{C_ERR}]"
    )
    console.print(f"  .env               {env_status}")

    ckpt = _PROJECT_ROOT / "models" / "enhanced" / "ppo_live_checkpoint.pt"
    if ckpt.exists():
        size_kb = ckpt.stat().st_size / 1024
        console.print(f"  PPO checkpoint     [{C_OK}]{size_kb:.0f} KB[/{C_OK}]")
    else:
        console.print(
            f"  PPO checkpoint     [{C_DIM}]None — created on first run[/{C_DIM}]"
        )

    console.print()


# ═══════════════════════════════════════════════════════════════════════
#  Advanced Configuration
# ═══════════════════════════════════════════════════════════════════════

def show_advanced_config() -> None:
    """Advanced configuration submenu — flags, PPO, routing, budget."""
    console.print()
    console.print(Rule(
        f"[{C_TITLE}]🛠️  Advanced Configuration[/{C_TITLE}]",
        style=C_ACCENT,
    ))

    choice = _pick("Category", "Select", [
        ("1", "🚩 Feature Flags",
         "View all 76+ runtime feature flags with env override status"),
        ("2", "🔀 Model Routing Table",
         "See how GPT model requests are routed by tier/provider"),
        ("3", "💰 Budget Manager Status",
         "Token budgets per tier, burst pool, usage stats"),
        ("4", "🧠 PPO Hyperparameters",
         "All PPO Actor-Critic v3.0 config values"),
        ("5", "🌍 Environment Variables",
         "All Ariaska/FF/MC/OPENAI env vars currently set"),
        ("b", "↩️  Back to Main Menu", ""),
    ], default="b")

    if choice == "b":
        return
    elif choice == "1":
        _show_feature_flags()
    elif choice == "2":
        _show_model_routing()
    elif choice == "3":
        _show_budget_status()
    elif choice == "4":
        _show_ppo_params()
    elif choice == "5":
        _show_env_vars()

    console.print()


def _show_feature_flags() -> None:
    """View all feature flags with env override status."""
    try:
        from core.feature_flags import FeatureFlags
        flags = FeatureFlags()

        table = Table(
            title="[bold]Active Feature Flags[/bold]",
            box=box.SIMPLE_HEAVY,
            border_style="bright_magenta",
            padding=(0, 1),
        )
        table.add_column("Flag", style="cyan", width=35)
        table.add_column("Value", width=8, justify="center")
        table.add_column("Env Override", style=C_DIM, width=20)

        for attr in sorted(dir(flags)):
            if attr.startswith("_") or attr == "profile":
                continue
            val = getattr(flags, attr, None)
            if isinstance(val, bool):
                val_str = (
                    f"[{C_OK}]ON[/{C_OK}]" if val else f"[{C_ERR}]OFF[/{C_ERR}]"
                )
                env_key = f"FF_{attr.upper()}"
                env_val = os.environ.get(env_key, "")
                override = (
                    f"[{C_WARN}]{env_val}[/{C_WARN}]"
                    if env_val else f"[{C_DIM}]—[/{C_DIM}]"
                )
                table.add_row(attr, val_str, override)
            elif isinstance(val, (int, float, str)) and not callable(val):
                table.add_row(attr, f"[white]{val}[/white]", f"[{C_DIM}]—[/{C_DIM}]")

        console.print(table)
        profile = getattr(flags, "profile", "unknown")
        console.print(
            f"\n[{C_DIM}]Profile: {profile} | "
            f"Override any flag: set FF_<FLAG_NAME>=0|1 in environment[/{C_DIM}]"
        )
    except Exception as e:
        console.print(f"[{C_ERR}]Error loading feature flags: {e}[/{C_ERR}]")


def _show_model_routing() -> None:
    """Show how model requests are routed."""
    try:
        from core.llm.model_router import ModelRouter
        router = ModelRouter.from_flags()

        table = Table(
            title="[bold]Model Routing Table[/bold]",
            box=box.SIMPLE_HEAVY,
            border_style="cyan",
        )
        table.add_column("Model", style="cyan", width=20)
        table.add_column("Tier", style=C_KEY, width=10)
        table.add_column("→ Provider", style=C_OK, width=30)

        test_models = [
            "gpt-5.2-nano", "gpt-5.2-mini", "gpt-5.2", "gpt-5.2-codex",
            "gpt-5-nano", "gpt-5-mini",
        ]
        for m in test_models:
            try:
                decision = router.route(m, "tactical")
                table.add_row(m, decision.tier, f"{decision.provider} ({decision.reason})")
            except Exception:
                table.add_row(m, "?", "routing error")

        console.print(table)
    except Exception as e:
        console.print(f"[{C_ERR}]Error loading model router: {e}[/{C_ERR}]")


def _show_budget_status() -> None:
    """Show BudgetManagerV2 status."""
    try:
        from core.llm.budget_manager import BudgetManagerV2
        bm = BudgetManagerV2()
        stats = bm.get_stats()
        console.print(Panel(
            json.dumps(stats, indent=2, default=str),
            title="[bold]Budget Manager Status[/bold]",
            border_style=C_ACCENT,
            box=box.ROUNDED,
        ))
    except Exception as e:
        console.print(f"[{C_ERR}]Error loading budget manager: {e}[/{C_ERR}]")


def _show_ppo_params() -> None:
    """Show PPO hyperparameters."""
    try:
        from core.algorithms.ppo_agent import PPOConfig
        config = PPOConfig()

        table = Table(
            title="[bold]PPO Actor-Critic v3.0 — Hyperparameters[/bold]",
            box=box.SIMPLE_HEAVY,
            border_style=C_ACCENT,
        )
        table.add_column("Parameter", style="cyan", width=30)
        table.add_column("Value", style="white", width=30)

        for attr in sorted(dir(config)):
            if not attr.startswith("_"):
                val = getattr(config, attr, None)
                if isinstance(val, (bool, int, float, str, list, tuple)):
                    table.add_row(attr, str(val))

        console.print(table)
    except Exception as e:
        console.print(f"[{C_ERR}]Error loading PPO config: {e}[/{C_ERR}]")


def _show_env_vars() -> None:
    """Show all Ariaska-related environment variables."""
    table = Table(
        title="[bold]Ariaska Environment Variables[/bold]",
        box=box.SIMPLE,
        border_style=C_DIM,
    )
    table.add_column("Variable", style="cyan", width=35)
    table.add_column("Value", style="white")

    ariaska_vars = sorted(
        (k, v) for k, v in os.environ.items()
        if k.startswith(("ARIASKA_", "FF_", "MC_", "OPENAI_", "MENTOR_", "VENICE_"))
    )
    for k, v in ariaska_vars:
        # Mask sensitive keys
        display_v = (v[:8] + "...") if "KEY" in k and v else v
        table.add_row(k, display_v)

    if not ariaska_vars:
        table.add_row(f"[{C_DIM}]None set[/{C_DIM}]", "")

    console.print(table)


# ═══════════════════════════════════════════════════════════════════════
#  Main Loop
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point — menu loop."""
    console.clear()
    console.print(BANNER)

    sys_info = detect_system()
    console.print(build_system_panel(sys_info))

    while True:
        try:
            choice = show_main_menu()

            if choice == "q":
                console.print(
                    f"\n[bold red]Ariaska offline.[/bold red]\n"
                )
                break

            elif choice == "1":
                cfg = wizard_launch(sys_info)
                if cfg:
                    execute(cfg)

            elif choice == "2":
                cfg = quick_launch(sys_info)
                if cfg:
                    execute(cfg)

            elif choice == "3":
                cfg = quick_ctf(sys_info)
                if cfg:
                    execute(cfg)

            elif choice == "4":
                cfg = quick_htb(sys_info)
                if cfg:
                    execute(cfg)

            elif choice == "5":
                show_training_history()

            elif choice == "6":
                resume_last_training()

            elif choice == "7":
                show_gpu_manager(sys_info)

            elif choice == "8":
                run_tests()

            elif choice == "9":
                show_diagnostics()

            elif choice == "0":
                show_advanced_config()

        except KeyboardInterrupt:
            console.print(
                f"\n[{C_WARN}]Interrupted — returning to menu.[/{C_WARN}]"
            )
            continue
        except EOFError:
            break


if __name__ == "__main__":
    main()
