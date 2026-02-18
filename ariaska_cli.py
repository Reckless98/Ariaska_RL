#!/usr/bin/env python3
"""
ARIASKA_RL Command Line Interface — Unified Entry Point (Phase 5)
=================================================================
Single entry point for all ARIASKA_RL operations:
  - smart-train: Full RL training pipeline with PPO, playbooks, Metasploitable 2 sim
  - status: System diagnostics
  - help: Usage info
"""

import sys
import os
import json
import time
import random
from pathlib import Path
from typing import Optional
from datetime import datetime

# ── Suppress noisy library output BEFORE any imports trigger loading ────────
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.path.join(str(Path.home()), ".cache", "torch", "sentence_transformers"))
import logging as _logging
for _noisy in ("sentence_transformers", "huggingface_hub", "transformers",
               "huggingface_hub.utils._http", "sentence_transformers.SentenceTransformer",
               "filelock"):
    _logging.getLogger(_noisy).setLevel(_logging.ERROR)
# Suppress tqdm progress bars globally (Batches: from sentence-transformers)
os.environ.setdefault("TQDM_DISABLE", "1")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
# Phase 6.9: Rich Progress REMOVED — caused carriage-return interleaving with logging
# from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich import box
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Suppress noisy HTTP retry logs (429 rate limit spam)
import logging as _logging
for _noisy in ("httpx", "openai", "openai._base_client", "httpcore"):
    _logging.getLogger(_noisy).setLevel(_logging.WARNING)

# Load environment variables
load_dotenv()

# Force Rich terminal rendering — ensures full Rich UI even if accidentally piped
console = Console(force_terminal=True)

# Ensure line buffering so output appears immediately
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# Deterministic Mode
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _init_deterministic_mode():
    """Initialize deterministic mode if enabled via environment variable."""
    if os.environ.get("ARIASKA_DETERMINISTIC", "").lower() == "true":
        seed = int(os.environ.get("ARIASKA_SEED", "42"))
        set_seed(seed)
        console.print(f"[green]🔒 Deterministic mode enabled (seed={seed})[/green]")
        return True
    return False

_deterministic_mode = _init_deterministic_mode()


# ─────────────────────────────────────────────────────────────────────────────
# SMART-TRAIN (consolidated from train_phase4.py)
# ─────────────────────────────────────────────────────────────────────────────
def run_training(
    max_steps: int = 500,
    seed: Optional[int] = None,
    target_ip: str = "10.129.1.54",
    mode: str = "live",
    platform: str = "linux",
    verbosity: str = "verbose",
    checkpoint_path: str = "models/enhanced/ppo_live_checkpoint.pt",
    dashboard_mode: str = "live",
    log_jsonl: bool = True,
    mentor_budget: Optional[float] = None,
    mentor_min_rate: float = 0.15,
    mentor_max_rate: float = 0.35,
    resume_path: Optional[str] = None,
    max_tokens_run: Optional[int] = None,
    checkpoint_every: int = 10,
    anti_forensics: bool = False,
    ethics_mode: str = "training",
    seed_skills: bool = False,
    ctf_mode: bool = False,
):
    """
    Single continuous engagement against target.
    Auto-closes when both user.txt and root.txt flags are captured.

    Args:
        max_steps: Safety limit on total steps (default 500).
        seed: Random seed (optional, None = non-deterministic).
        target_ip: Target IP (REQUIRED).
        mode: Always 'live'.
        platform: Target platform hint.
        verbosity: 'quiet', 'standard', or 'verbose'.
        checkpoint_path: Path for PPO checkpoint save/load.
        ctf_mode: If True, auto-close immediately on flag capture.

    Returns:
        Dictionary with engagement results and flag capture state.
    """
    if seed is not None:
        set_seed(seed)

    from core.orchestration.smart_orchestrator import SmartOrchestrator, SmartOrchestratorConfig
    from core.environment.cyber_environment import CyberEnvironment
    from core.gpt_manager import GPTManager

    # Phase 17: Auto-detect profile and enable LLM flags if API key present
    from core.feature_flags import resolve_profile
    _profile = resolve_profile()

    # Environment
    env = CyberEnvironment(defer_reset=False)
    env.mode = mode
    env.target_ip = target_ip
    if mode == "live":
        env.live_mode = True
        env.live_target_ip = target_ip

    gpt = GPTManager.get_instance()

    config = SmartOrchestratorConfig(
        model="gpt-5.2-codex",
        mentor_mode="adaptive",
        mentor_warmup_episodes=2,
        mentor_min_rate=mentor_min_rate,
        mentor_max_rate=mentor_max_rate,
        mentor_budget_pct=mentor_budget if mentor_budget is not None else 1.0,  # Phase 36: adaptive default, explicit overrides
        max_steps_per_episode=max_steps,  # Safety limit for continuous run
        default_target=target_ip,
        dashboard_enabled=(verbosity != "quiet"),
        dashboard_mode="live" if dashboard_mode in ("rich", "live") else dashboard_mode,
        event_jsonl_path=(
            f"traces/events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            if log_jsonl else None
        ),
        difficulty="normal",
        ctf_mode=ctf_mode,
    )

    orch = SmartOrchestrator(
        env=env,
        gpt_manager=gpt,
        config=config,
        verbosity=verbosity,
    )

    # Load checkpoint if exists
    if resume_path and os.path.exists(resume_path):
        orch.load_ppo_checkpoints(resume_path)
    elif os.path.exists(checkpoint_path):
        orch.load_ppo_checkpoints(checkpoint_path)
    
    # Phase 6.2: Initialize CheckpointManager for atomic saves
    from core.training.checkpoint_manager import CheckpointManager, CheckpointConfig
    ckpt_mgr = CheckpointManager(config=CheckpointConfig(
        checkpoint_dir=os.path.dirname(checkpoint_path) or "models/enhanced",
        checkpoint_name=os.path.basename(checkpoint_path),
        auto_save_interval=50,  # Auto-save every 50 steps in continuous mode
        target_health_enabled=(mode == "live"),
        target_ip=target_ip if mode == "live" else "",
    ))

    # ── Training Loop ────────────────────────────────────────────────────
    start_time = time.time()

    # Phase 6.6: Initialize overnight guardrails
    from core.training.guardrails import (
        GuardrailManager, BudgetGovernor, HealthChecker,
        StagnationWatchdog, CheckpointScheduler, LogRotator,
    )
    guardrails = GuardrailManager(
        budget=BudgetGovernor(
            max_tokens_per_run=max_tokens_run,
        ),
        health=HealthChecker(
            target_ip=target_ip if mode == "live" else "",
            enabled=(mode == "live"),
        ),
        watchdog=StagnationWatchdog(),
        checkpoint=CheckpointScheduler(
            interval=checkpoint_every,
        ),
        log_rotator=LogRotator(),
    )

    # Phase 25: Banner is printed once by SmartOrchestrator.run_training()
    # with full GPT model info — no duplicate banner here.

    # Phase 6.7: Pre-seed SkillLibrary with expert knowledge
    if seed_skills:
        try:
            from core.postmortem.skill_library import SkillLibrary
            skill_lib = SkillLibrary()
            count = skill_lib.seed_skills()
        except Exception as e:
            console.print(f"[yellow]⚠️ Skill seeding failed: {e}[/yellow]")

    # Phase 6.7: Store anti-forensics config on orchestrator
    orch.anti_forensics_enabled = anti_forensics  # type: ignore[attr-defined]
    orch.ethics_mode = ethics_mode  # type: ignore[attr-defined]

    # Phase 10.3: Suppress noisy INFO loggers during training — all relevant
    # information is shown via Rich dashboard panels. Raw logger.info() lines
    # between steps (TacticalCortex, MENTOR-REASONING, PHASE-GATE, GPT empty
    # response warnings) break the clean UI flow. We raise them to WARNING+
    # selectively and restore after training.
    _noisy_loggers = [
        "ariaska.smart_coach", "ariaska.smart_orchestrator",
        "core.gpt_manager", "ariaska.trainer", "ariaska.parser_broker",
        "ariaska.ppo", "ariaska.ddqn_macro", "ariaska.cognition_node",
        "ariaska.skill_library", "ariaska.reward_calculator",
        "ariaska.telemetry",
    ]
    _saved_levels = {}
    for _lg_name in _noisy_loggers:
        _lg = _logging.getLogger(_lg_name)
        _saved_levels[_lg_name] = _lg.level
        # core.gpt_manager emits WARNING-level noise (empty GPT responses),
        # so raise it to ERROR; raise all others to WARNING.
        _lg.setLevel(_logging.ERROR if _lg_name == "core.gpt_manager" else _logging.WARNING)

    # ── Single Continuous Engagement — orchestrator handles everything ──
    # run_training() forces episodes=1 internally and runs until flag capture
    # or max_steps safety limit. Auto-close triggers on user.txt + root.txt.
    try:
        result = orch.run_training(
            episodes=1,
            target_ip=target_ip,
            platform=platform,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Engagement interrupted by user[/yellow]")
        result = {}

    elapsed = time.time() - start_time

    # Phase 10.3: Restore logger levels after training
    for _lg_name, _orig_level in _saved_levels.items():
        _logging.getLogger(_lg_name).setLevel(_orig_level)

    # Save PPO checkpoint
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    orch.save_ppo_checkpoints(checkpoint_path)

    # ── Summary ──────────────────────────────────────────────────────────
    if not result:
        console.print("[yellow]⚠️ No results — engagement may have been interrupted.[/yellow]")
        return {"result": {}, "duration_s": elapsed}

    ep_metrics = result.get("episode_metrics", {})
    reward = ep_metrics.get("total_reward", result.get("total_reward", 0.0))
    highest = ep_metrics.get("highest_phase", result.get("highest_phase", "RECON"))
    steps = ep_metrics.get("total_steps", result.get("total_steps", 0))
    user_flag = result.get("user_flag_captured", False)
    root_flag = result.get("root_flag_captured", False)
    user_val = result.get("user_flag_value", "")
    root_val = result.get("root_flag_value", "")

    summary_table = Table(title="📊 ARIASKA Engagement Summary", box=box.ROUNDED)
    summary_table.add_column("Metric", style="cyan", width=24)
    summary_table.add_column("Value", style="bold white", width=40)
    summary_table.add_column("Quality", style="dim", width=16)
    summary_table.add_row("Mode", "CONTINUOUS", "")
    summary_table.add_row("Target", target_ip, "")
    summary_table.add_row("Total Steps", str(steps), f"limit: {max_steps}")
    summary_table.add_row("Duration", f"{elapsed:.1f}s ({elapsed/max(steps,1):.2f}s/step)", "")
    summary_table.add_row("Total Reward", f"[bold green]{reward:+.1f}[/bold green]",
                          f"{'🔥 excellent' if reward > 1200 else '✅ good' if reward > 800 else '⚠️ moderate'}")
    summary_table.add_row("Highest Phase", f"[bold yellow]{highest}[/bold yellow]", "")
    summary_table.add_row("─" * 24, "─" * 40, "─" * 16)
    summary_table.add_row("🏴 user.txt",
                          f"[bold green]CAPTURED: {user_val}[/bold green]" if user_flag else "[red]NOT FOUND[/red]",
                          "🏆" if user_flag else "")
    summary_table.add_row("🏴 root.txt",
                          f"[bold green]CAPTURED: {root_val}[/bold green]" if root_flag else "[red]NOT FOUND[/red]",
                          "🏆" if root_flag else "")

    if user_flag and root_flag:
        summary_table.add_row("─" * 24, "─" * 40, "─" * 16)
        summary_table.add_row("RESULT", "[bold green]COMPLETE PWN 🎉[/bold green]", "🏆🏆🏆")

    # Reward-invariant metrics
    summary_table.add_row("─" * 24, "─" * 40, "─" * 16)
    summary_table.add_row("Unique Commands", str(ep_metrics.get("unique_commands_total", 0)), "")
    summary_table.add_row("Command Diversity", f"{ep_metrics.get('command_diversity_ratio', 0.0):.2f}", "")
    summary_table.add_row("Total Discoveries", str(ep_metrics.get("total_discoveries", 0)), "")
    summary_table.add_row("First Exploit Step", str(ep_metrics.get("step_at_first_exploit", -1)), "")
    console.print(summary_table)

    # Decision source summary
    sources = {
        "ppo": ep_metrics.get("decisions_ppo", result.get("decisions_ppo", 0)),
        "playbook": ep_metrics.get("decisions_playbook", result.get("decisions_playbook", 0)),
        "registry": ep_metrics.get("decisions_registry", result.get("decisions_registry", 0)),
        "anti_repeat": ep_metrics.get("decisions_anti_repeat", result.get("decisions_anti_repeat", 0)),
        "codex_meta": ep_metrics.get("decisions_codex_meta", result.get("decisions_codex_meta", 0)),
    }
    total_decisions = sum(sources.values()) or 1
    if any(sources.values()):
        src_table = Table(title="🎯 Decision Sources", box=box.SIMPLE)
        src_table.add_column("Source", style="cyan")
        src_table.add_column("Count", style="white")
        src_table.add_column("Percent", style="yellow")
        for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
            if cnt > 0:
                src_table.add_row(src, str(cnt), f"{cnt/total_decisions*100:.1f}%")
        console.print(src_table)

    # Save results
    results_path = f"artifacts/engagement_{datetime.now().strftime('%Y%m%d_%H%M%S')}_results.json"
    os.makedirs("artifacts", exist_ok=True)
    results = {
        "timestamp": datetime.now().isoformat(),
        "max_steps": max_steps,
        "seed": seed,
        "target_ip": target_ip,
        "mode": "continuous",
        "duration_s": elapsed,
        "total_reward": reward,
        "highest_phase": highest,
        "total_steps": steps,
        "user_flag_captured": user_flag,
        "root_flag_captured": root_flag,
        "user_flag_value": user_val,
        "root_flag_value": root_val,
        "decision_sources": sources,
        "episode_metrics": ep_metrics,
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    console.print(f"\n[green]💾 Results saved to {results_path}[/green]")
    console.print(f"[green]💾 PPO checkpoint saved to {checkpoint_path}[/green]")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# STATUS
# ─────────────────────────────────────────────────────────────────────────────
def show_system_status():
    """Show system status and diagnostics."""
    console.print(Panel("🔍 System Status Check", title="ARIASKA_RL Diagnostics", border_style="blue"))

    status_table = Table(show_header=True, header_style="bold cyan")
    status_table.add_column("Component", style="white", width=30)
    status_table.add_column("Status", style="green", width=15)
    status_table.add_column("Details", style="yellow", width=40)

    env_ok = Path(".env").exists()
    status_table.add_row(".env Configuration", "✅ Found" if env_ok else "❌ Missing", "Environment variables")

    core_checks = [
        ("SmartOrchestrator", "core/orchestration/smart_orchestrator.py"),
        ("SmartCoach", "core/training/smart_coach.py"),
        ("CommandRegistry", "core/commands/command_registry.py"),
        ("RewardCalculator", "core/llm/reward_calculator.py"),
        ("GPTManager", "core/gpt_manager.py"),
    ]
    for name, path in core_checks:
        ok = Path(path).exists()
        status_table.add_row(name, "✅ Found" if ok else "❌ Missing", path)

    ckpt = Path("models/enhanced/ppo_checkpoint.pt")
    if ckpt.exists():
        size_kb = ckpt.stat().st_size / 1024
        status_table.add_row("PPO Checkpoint", "✅ Found", f"{size_kb:.0f} KB")
    else:
        status_table.add_row("PPO Checkpoint", "⚠️ None", "Will be created on first train")

    log_dir = Path("logs")
    if log_dir.exists():
        n = len(list(log_dir.rglob("*")))
        status_table.add_row("Logs", "✅ Found", f"{n} files")
    else:
        status_table.add_row("Logs", "⚠️ Empty", "")

    console.print(status_table)

    env_table = Table(show_header=True, header_style="bold blue")
    env_table.add_column("Setting", style="cyan")
    env_table.add_column("Value", style="white")
    env_table.add_row("PRIMARY_MODEL", os.getenv("PRIMARY_MODEL", "Not set"))
    env_table.add_row("ARIASKA_MODE", os.getenv("ARIASKA_MODE", "Not set"))
    env_table.add_row("ARIASKA_TARGET_IP", os.getenv("ARIASKA_TARGET_IP", "Not set"))
    env_table.add_row("ARIASKA_DETERMINISTIC", os.getenv("ARIASKA_DETERMINISTIC", "false"))
    console.print(Panel(env_table, title="Environment Configuration"))


# ─────────────────────────────────────────────────────────────────────────────
# HELP
# ─────────────────────────────────────────────────────────────────────────────
def show_help():
    """Display help information."""
    help_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    help_table.add_column("Command", style="cyan", width=45)
    help_table.add_column("Description", style="white", width=35)
    help_table.add_column("Example", style="green", width=35)

    help_table.add_row(
        "smart-train [OPTIONS]",
        "🧠 Continuous engagement with\nPPO, auto-close on flag capture",
        "ariaska smart-train --target IP",
    )
    help_table.add_row("  --steps N", "Max steps safety limit (default 500)", "--steps 1000")
    help_table.add_row("  --seed N", "Random seed (optional)", "--seed 123")
    help_table.add_row("  --target IP", "Target IP (REQUIRED)", "--target 10.129.1.54")
    help_table.add_row("  --ctf", "CTF mode: close on flag capture", "--ctf")
    help_table.add_row("  --verbosity LEVEL", "quiet / standard / verbose", "--verbosity verbose")
    help_table.add_row("replay <TRACE>", "Replay a JSONL event trace file", "ariaska replay traces/events_*.jsonl")
    help_table.add_row("  --verbose", "Show stdout snippets in replay", "--verbose")
    help_table.add_row(
        "watchdog [OPTIONS]",
        "🐕 Persistent runner: run, monitor,\ndebug, relaunch until both flags",
        "ariaska watchdog --target IP",
    )
    help_table.add_row("  --max-attempts N", "Max restart attempts (default: 10)", "--max-attempts 20")
    help_table.add_row("status", "Show system status & diagnostics", "ariaska status")
    help_table.add_row("help", "Show this help message", "ariaska help")

    console.print(Panel(
        help_table,
        title="🧠 ARIASKA_RL — Autonomous Cybersecurity RL",
        subtitle="Continuous Engagement | Auto-Close on Flag Capture",
        border_style="blue",
    ))


# ─────────────────────────────────────────────────────────────────────────────
# P34-EXT: WATCHDOG RUNNER — persistent engagement until both flags
# ─────────────────────────────────────────────────────────────────────────────
def _run_watchdog(args):
    """
    Persistent watchdog runner: run engagement, monitor for stagnation,
    auto-debug, and relaunch until both user.txt and root.txt are captured
    or max-attempts is exhausted.

    Stagnation detection:
      - stagnation_steps >= 8 (no new discoveries for 8+ steps)
      - anti_repeat_rate >= 30% (3/10 steps blocked)
      - avg confidence < 0.55

    On stagnation: log diagnosis to watchdog trace, restart with new seed.
    """
    target_ip = args.target
    max_steps = args.steps
    max_attempts = args.max_attempts
    seed = args.seed
    verbosity = args.verbosity
    ctf_mode = args.ctf

    # Watchdog trace file
    os.makedirs("traces", exist_ok=True)
    trace_path = f"traces/watchdog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    console.print(Panel(
        f"[bold]Target:[/bold] {target_ip}\n"
        f"[bold]Steps/attempt:[/bold] {max_steps}\n"
        f"[bold]Max attempts:[/bold] {max_attempts}\n"
        f"[bold]CTF mode:[/bold] {'ON' if ctf_mode else 'OFF'}\n"
        f"[bold]Trace:[/bold] {trace_path}\n"
        f"\n[dim]Will auto-restart on stagnation. Ctrl+C to stop.[/dim]",
        title="[bold bright_cyan]🐕 ARIASKA Watchdog Runner[/bold bright_cyan]",
        border_style="bright_cyan",
        box=box.ROUNDED,
    ))

    user_flag = False
    root_flag = False
    attempt = 0

    while attempt < max_attempts and not (user_flag and root_flag):
        attempt += 1
        attempt_seed = (seed + attempt) if seed is not None else None
        attempt_start = time.time()

        console.print(f"\n[bold bright_cyan]🐕 Watchdog: Attempt {attempt}/{max_attempts}[/bold bright_cyan]")

        # Log attempt start
        _watchdog_log(trace_path, {
            "event": "attempt_start",
            "attempt": attempt, "seed": attempt_seed,
            "target": target_ip, "max_steps": max_steps,
            "timestamp": datetime.now().isoformat(),
        })

        try:
            results = run_training(
                max_steps=max_steps,
                seed=attempt_seed,
                target_ip=target_ip,
                mode="live",
                platform="linux",
                verbosity=verbosity,
                ctf_mode=ctf_mode,
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️ Watchdog interrupted by user[/yellow]")
            _watchdog_log(trace_path, {
                "event": "user_interrupt", "attempt": attempt,
                "timestamp": datetime.now().isoformat(),
            })
            break
        except Exception as e:
            console.print(f"[red]❌ Attempt {attempt} crashed: {e}[/red]")
            _watchdog_log(trace_path, {
                "event": "attempt_crash", "attempt": attempt,
                "error": str(e), "timestamp": datetime.now().isoformat(),
            })
            time.sleep(5)  # Cooldown before restart
            continue

        elapsed = time.time() - attempt_start

        # Check results
        if results:
            user_flag = results.get("user_flag_captured", False) or user_flag
            root_flag = results.get("root_flag_captured", False) or root_flag
            ep_metrics = results.get("episode_metrics", {})
            highest = ep_metrics.get("highest_phase", results.get("highest_phase", "RECON"))
            reward = ep_metrics.get("total_reward", results.get("total_reward", 0.0))
            steps = ep_metrics.get("total_steps", results.get("total_steps", 0))
            unique_cmds = ep_metrics.get("unique_commands_total", 0)
            discoveries = ep_metrics.get("total_discoveries", 0)

            _watchdog_log(trace_path, {
                "event": "attempt_complete", "attempt": attempt,
                "elapsed_s": round(elapsed, 1),
                "reward": round(reward, 1), "highest_phase": highest,
                "steps": steps, "unique_cmds": unique_cmds,
                "discoveries": discoveries,
                "user_flag": user_flag, "root_flag": root_flag,
                "user_flag_value": results.get("user_flag_value", ""),
                "root_flag_value": results.get("root_flag_value", ""),
                "timestamp": datetime.now().isoformat(),
            })

            # Stagnation diagnosis
            lm = ep_metrics.get("learning_metrics", {})
            max_stag = lm.get("max_stagnation", 0)
            ar_rate = lm.get("anti_repeat_rate", 0.0)
            novelty = lm.get("novelty_rate", 0.0)

            stagnation_detected = (
                max_stag >= 8
                or ar_rate >= 0.30
                or (novelty < 0.25 and steps > 20)
            )

            if stagnation_detected and not (user_flag and root_flag):
                diagnosis = (
                    f"stag={max_stag}, AR={ar_rate:.0%}, novelty={novelty:.0%}, "
                    f"phase={highest}, steps={steps}"
                )
                console.print(Panel(
                    f"[yellow]Stagnation detected:[/yellow] {diagnosis}\n"
                    f"[dim]Auto-restarting with new seed...[/dim]",
                    title="[bold yellow]🐕 Watchdog: Stagnation Detected[/bold yellow]",
                    border_style="yellow", box=box.ROUNDED,
                ))
                _watchdog_log(trace_path, {
                    "event": "stagnation_detected", "attempt": attempt,
                    "diagnosis": diagnosis, "timestamp": datetime.now().isoformat(),
                })
                time.sleep(3)  # Brief cooldown
                continue

        if user_flag and root_flag:
            console.print(Panel(
                f"[bold green]🏆 BOTH FLAGS CAPTURED! 🏆[/bold green]\n\n"
                f"User: {results.get('user_flag_value', '')}\n"
                f"Root: {results.get('root_flag_value', '')}\n\n"
                f"[dim]Attempts: {attempt} │ Total time: {elapsed:.0f}s[/dim]",
                title="[bold green]🐕 Watchdog: MISSION COMPLETE[/bold green]",
                border_style="bold green", box=box.ROUNDED,
            ))
            _watchdog_log(trace_path, {
                "event": "mission_complete", "attempt": attempt,
                "timestamp": datetime.now().isoformat(),
            })
            break

    if not (user_flag and root_flag):
        console.print(Panel(
            f"[yellow]Exhausted {max_attempts} attempts without full pwn.[/yellow]\n"
            f"User flag: {'✅' if user_flag else '❌'}\n"
            f"Root flag: {'✅' if root_flag else '❌'}\n\n"
            f"[dim]Review: {trace_path}[/dim]",
            title="[bold yellow]🐕 Watchdog: Max Attempts Reached[/bold yellow]",
            border_style="yellow", box=box.ROUNDED,
        ))

    console.print(f"\n[dim]📋 Watchdog trace: {trace_path}[/dim]")
    console.print(f"[dim]💡 Replay: ariaska replay <trace_file>[/dim]")


def _watchdog_log(path: str, data: dict) -> None:
    """Append a JSONL record to the watchdog trace."""
    try:
        with open(path, "a") as f:
            f.write(json.dumps(data, default=str) + "\n")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="ariaska",
        description="ARIASKA_RL — Cybersecurity Reinforcement Learning",
    )
    sub = parser.add_subparsers(dest="command")

    # smart-train
    train_p = sub.add_parser("smart-train", help="Run continuous engagement against target")
    train_p.add_argument("--steps", "-s", type=int, default=500, help="Max steps safety limit (default: 500)")
    train_p.add_argument("--seed", type=int, default=None, help="Random seed (optional, non-deterministic by default)")
    train_p.add_argument("--target", type=str, required=True, help="Target IP (REQUIRED)")
    train_p.add_argument("--ctf", action="store_true", default=False,
                         help="CTF mode: auto-close on user.txt + root.txt flag capture (for HTB/CTF machines)")
    train_p.add_argument("--verbosity", "-v", type=str, default="verbose",
                         choices=["quiet", "standard", "verbose"])
    train_p.add_argument("--checkpoint", type=str, default=None,
                         help="PPO checkpoint path (auto-selected per mode if omitted)")
    
    # Phase 6.2: New flags
    train_p.add_argument("--dashboard", type=str, default="live",
                         choices=["live", "textual", "rich", "off"],
                         help="Dashboard mode (default: live for Rich step panels)")
    train_p.add_argument("--log-jsonl", action="store_true", default=True,
                         help="Enable JSONL event logging (default: on)")
    train_p.add_argument("--no-log-jsonl", action="store_true", default=False,
                         help="Disable JSONL event logging")
    train_p.add_argument("--mentor-budget", type=float, default=None,
                         help="Override adaptive mentor budget with fixed fraction (default: adaptive)")
    train_p.add_argument("--mentor-min-rate", type=float, default=0.15,
                         help="Minimum mentor call rate (default: 0.15)")
    train_p.add_argument("--mentor-max-rate", type=float, default=0.35,
                         help="Maximum mentor call rate (default: 0.35)")
    train_p.add_argument("--resume", type=str, default=None,
                         help="Resume from checkpoint path")
    
    train_p.add_argument("--max-tokens-run", type=int, default=None,
                         help="Total token budget for the entire run (default: unlimited)")
    train_p.add_argument("--checkpoint-every", type=int, default=10,
                         help="Save PPO checkpoints every N episodes (default: 10)")

    # Phase 6.7: Anti-forensics & ethics mode
    train_p.add_argument("--anti-forensics", action="store_true", default=False,
                         help="Enable anti-forensics in CLOSEOUT phase (default: OFF)")
    train_p.add_argument("--no-anti-forensics", dest="anti_forensics", action="store_false",
                         help="Disable anti-forensics in CLOSEOUT phase")
    train_p.add_argument("--ethics-mode", type=str, default="training",
                         choices=["training", "assessment", "demo"],
                         help="Ethics mode: training=full capability, assessment=audit logging, demo=investor-safe display (default: training)")
    train_p.add_argument("--seed-skills", action="store_true", default=True,
                         help="Pre-seed SkillLibrary with expert skill cards on startup (default: ON)")
    train_p.add_argument("--no-seed-skills", dest="seed_skills", action="store_false",
                         help="Disable pre-seeding of SkillLibrary")

    # Phase 11.0: Full Visibility & Step Discipline
    train_p.add_argument("--parser-mode", type=str, default="intelligent_fullparse",
                         choices=["fast", "intelligent_fullparse"],
                         help="Parser mode: fast=regex only, intelligent_fullparse=4-stage cascade (default: intelligent_fullparse)")
    train_p.add_argument("--strict-ladder", action="store_true", default=True,
                         help="Strict phase ladder enforcement (default: ON)")
    train_p.add_argument("--no-strict-ladder", dest="strict_ladder", action="store_false",
                         help="Disable strict phase ladder enforcement")
    train_p.add_argument("--adaptive-budget", action="store_true", default=True,
                         help="Enable adaptive LLM budget controller (default: ON)")
    train_p.add_argument("--no-adaptive-budget", dest="adaptive_budget", action="store_false",
                         help="Disable adaptive budget controller")
    train_p.add_argument("--learning-signals", action="store_true", default=True,
                         help="Export learning signal JSONL (default: ON)")
    train_p.add_argument("--no-learning-signals", dest="learning_signals", action="store_false",
                         help="Disable learning signal export")

    # Legacy positional support removed — continuous mode uses --steps only
    train_p.add_argument("pos_env", nargs="?", type=str, default=None)

    # Phase 7.1: ingest-htb removed — knowledge is now hardcoded in knowledge_packs.py

    # Phase 31: Replay past engagements from JSONL trace files
    replay_p = sub.add_parser("replay", help="Replay a JSONL event trace file")
    replay_p.add_argument("trace_file", type=str, help="Path to the JSONL trace file (e.g. traces/events_*.jsonl)")
    replay_p.add_argument("--verbose", action="store_true", default=False,
                          help="Show stdout output snippets for each agent action")

    # P34-EXT: Watchdog runner — persistent engagement until both flags
    watchdog_p = sub.add_parser("watchdog", help="Persistent watchdog: run, monitor, debug, relaunch until both flags")
    watchdog_p.add_argument("--target", type=str, required=True, help="Target IP (REQUIRED)")
    watchdog_p.add_argument("--steps", "-s", type=int, default=500, help="Max steps per attempt (default: 500)")
    watchdog_p.add_argument("--max-attempts", type=int, default=10, help="Max restart attempts (default: 10)")
    watchdog_p.add_argument("--seed", type=int, default=None, help="Random seed")
    watchdog_p.add_argument("--ctf", action="store_true", default=True, help="CTF mode (default: ON)")
    watchdog_p.add_argument("--verbosity", "-v", type=str, default="verbose",
                            choices=["quiet", "standard", "verbose"])

    sub.add_parser("status", help="Show system status")
    sub.add_parser("help", help="Show help")

    args = parser.parse_args()

    if args.command is None or args.command == "help":
        show_help()
        return

    if args.command == "status":
        show_system_status()
        return

    if args.command == "replay":
        from core.replay.episode_replayer import replay_trace_file
        try:
            replay_trace_file(args.trace_file, verbose=args.verbose, console=console)
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            return 1
        except Exception as e:
            console.print(f"[red]Replay failed: {e}[/red]")
            return 1
        return

    if args.command == "watchdog":
        _run_watchdog(args)
        return

    if args.command == "smart-train":
        # ─── Continuous Engagement — single run until flag capture ───
        target_ip = args.target  # Required argument
        checkpoint_path = args.checkpoint or "models/enhanced/ppo_live_checkpoint.pt"

        os.environ["ARIASKA_MODE"] = "live"
        os.environ["ARIASKA_LIVE_MODE"] = "true"
        os.environ["ARIASKA_TARGET_IP"] = target_ip

        # Phase 11.0: Apply CLI flags to feature flags
        try:
            from core.feature_flags import get_feature_flags, set_feature_flag
            set_feature_flag("parser_mode", args.parser_mode)
            set_feature_flag("strict_phase_ladder", args.strict_ladder)
            set_feature_flag("adaptive_budget", args.adaptive_budget)
            set_feature_flag("learning_signal_export", args.learning_signals)
        except Exception:
            pass  # Feature flags module may not be loaded yet

        try:
            results = run_training(
                max_steps=args.steps,
                seed=args.seed,
                target_ip=target_ip,
                mode="live",
                platform="linux",
                verbosity=args.verbosity,
                checkpoint_path=checkpoint_path,
                dashboard_mode=args.dashboard,
                log_jsonl=args.log_jsonl and not args.no_log_jsonl,
                mentor_budget=args.mentor_budget,
                mentor_min_rate=args.mentor_min_rate,
                mentor_max_rate=args.mentor_max_rate,
                resume_path=args.resume,
                max_tokens_run=args.max_tokens_run,
                checkpoint_every=args.checkpoint_every,
                anti_forensics=args.anti_forensics,
                ethics_mode=args.ethics_mode,
                seed_skills=args.seed_skills,
                ctf_mode=args.ctf,
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️ Training interrupted by user[/yellow]")
        except Exception as e:
            console.print(f"[red]❌ Training failed: {e}[/red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return 1


if __name__ == "__main__":
    main()
