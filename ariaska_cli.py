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
from collections import Counter
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
console = Console(force_terminal=True, width=140)

# Ensure line buffering so output appears immediately
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(line_buffering=True)
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
    episodes: int = 100,
    max_steps: int = 40,
    seed: int = 42,
    target_ip: str = "172.28.0.10",
    mode: str = "live",
    platform: str = "linux",
    difficulty: str = "easy",
    verbosity: str = "verbose",
    checkpoint_path: str = "models/enhanced/ppo_checkpoint.pt",
    dashboard_mode: str = "rich",
    log_jsonl: bool = True,
    mentor_budget: float = 0.30,
    mentor_min_rate: float = 0.15,
    mentor_max_rate: float = 0.35,
    resume_path: Optional[str] = None,
    difficulty_preset: str = "normal",
    max_tokens_run: Optional[int] = None,
    checkpoint_every: int = 10,
    anti_forensics: bool = True,
    ethics_mode: str = "training",
    seed_skills: bool = False,
):
    """
    Consolidated training loop with PPO metrics, checkpoint persistence,
    and rich reporting.

    Args:
        episodes: Number of training episodes.
        max_steps: Maximum steps per episode.
        seed: Random seed for reproducibility.
        target_ip: Target IP for the environment.
        mode: 'simulated' or 'live'.
        platform: Target platform hint.
        difficulty: Difficulty preset.
        verbosity: 'quiet', 'standard', or 'verbose'.
        checkpoint_path: Path for PPO checkpoint save/load.

    Returns:
        Dictionary with full training results.
    """
    set_seed(seed)

    from core.orchestration.smart_orchestrator import SmartOrchestrator, SmartOrchestratorConfig
    from core.environment.cyber_environment import CyberEnvironment
    from core.gpt_manager import GPTManager

    # Environment
    env = CyberEnvironment(defer_reset=False)
    env.mode = mode
    env.target_ip = target_ip
    if mode == "live":
        env.live_mode = True
        env.live_target_ip = target_ip

    gpt = GPTManager()

    config = SmartOrchestratorConfig(
        model="gpt-5.1-codex-mini",
        mentor_mode="adaptive",
        mentor_warmup_episodes=2,
        mentor_min_rate=mentor_min_rate,
        mentor_max_rate=mentor_max_rate,
        mentor_budget_pct=mentor_budget,
        max_steps_per_episode=max_steps,
        default_target=target_ip,
        dashboard_enabled=(verbosity != "quiet"),
        dashboard_mode="live" if dashboard_mode == "rich" else dashboard_mode,
        event_jsonl_path=(
            f"traces/events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            if log_jsonl else None
        ),
        difficulty=difficulty_preset,
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
        console.print(f"[green]✅ Resumed from checkpoint: {resume_path}[/green]")
    elif os.path.exists(checkpoint_path):
        orch.load_ppo_checkpoints(checkpoint_path)
        console.print(f"[green]✅ Loaded PPO checkpoint from {checkpoint_path}[/green]")
    
    # Phase 6.2: Initialize CheckpointManager for atomic saves
    from core.training.checkpoint_manager import CheckpointManager, CheckpointConfig
    ckpt_mgr = CheckpointManager(config=CheckpointConfig(
        checkpoint_dir=os.path.dirname(checkpoint_path) or "models/enhanced",
        checkpoint_name=os.path.basename(checkpoint_path),
        auto_save_interval=5,
        target_health_enabled=(mode == "live"),
        target_ip=target_ip if mode == "live" else "",
    ))

    # ── Training Loop ────────────────────────────────────────────────────
    all_rewards = []
    all_phases = []
    episode_data = []
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

    console.print(Panel(
        f"[bold cyan]ARIASKA Smart Training v6.9 — CLOSEOUT Enforced[/bold cyan]\n\n"
        f"Episodes: {episodes}  |  Steps/ep: {max_steps}  |  Seed: {seed}\n"
        f"Target: {target_ip}  |  Mode: {mode.upper()}  |  Difficulty: {difficulty_preset.upper()}\n"
        f"Anti-forensics: {'[green]ON[/green]' if anti_forensics else '[red]OFF[/red]'}  |  Ethics: {ethics_mode.upper()}\n"
        f"Verbosity: {verbosity}  |  Checkpoint: {checkpoint_path}",
        title="🚀 Training Start",
        border_style="cyan",
    ))

    # Phase 6.7: Pre-seed SkillLibrary with expert knowledge
    if seed_skills:
        try:
            from core.postmortem.skill_library import SkillLibrary
            skill_lib = SkillLibrary()
            count = skill_lib.seed_skills()
            console.print(f"[green]🧠 Pre-seeded {count} expert skill cards into SkillLibrary[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Skill seeding failed: {e}[/yellow]")

    # Phase 6.7: Store anti-forensics config on orchestrator
    orch.anti_forensics_enabled = anti_forensics
    orch.ethics_mode = ethics_mode

    # Phase 6.9: NO Rich Progress wrapper — it causes carriage-return interleaving
    # with Python logging, producing the "glued monster line" UI bug.
    # Instead: simple console.print after each episode. Clean, visible, no overlap.
    for ep in range(episodes):
        # Phase 6.6: Pre-episode guardrail check
        if not guardrails.pre_episode_check(ep):
            console.print(f"[yellow]⚠️ Guardrails stopped training at episode {ep}[/yellow]")
            break
        
        episode_id = f"phase5_ep{ep:04d}"
        ep_result = orch.run_episode(
            episode_id=episode_id,
            episode_number=ep,
            target=target_ip,
            difficulty=difficulty,
            platform=platform,
        )

        reward = ep_result.get("total_reward", 0.0)
        highest = ep_result.get("highest_phase", "RECON")
        steps = ep_result.get("total_steps", 0)
        all_rewards.append(reward)
        all_phases.append(highest)

        ppo_metrics = {
            "updates_fired": ep_result.get("ppo_updates_fired", 0),
            "avg_policy_loss": ep_result.get("ppo_avg_policy_loss", 0.0),
            "avg_value_loss": ep_result.get("ppo_avg_value_loss", 0.0),
            "avg_entropy": ep_result.get("ppo_avg_entropy", 0.0),
        }
        sources = {
            "ppo": ep_result.get("decisions_ppo", 0),
            "playbook": ep_result.get("decisions_playbook", 0),
            "registry": ep_result.get("decisions_registry", 0),
            "anti_repeat": ep_result.get("decisions_anti_repeat", 0),
        }

        ep_info = {
            "episode": ep + 1,
            "reward": reward,
            "highest_phase": highest,
            "steps": steps,
            "ppo_updates": ppo_metrics.get("updates_fired", 0),
            "policy_loss": ppo_metrics.get("avg_policy_loss", 0.0),
            "value_loss": ppo_metrics.get("avg_value_loss", 0.0),
            "entropy": ppo_metrics.get("avg_entropy", 0.0),
            "sources": sources,
            # Phase 5.1: reward-invariant metrics
            "unique_commands": ep_result.get("unique_commands_total", 0),
            "unique_templates": ep_result.get("unique_templates_total", 0),
            "command_diversity": ep_result.get("command_diversity_ratio", 0.0),
            "total_discoveries": ep_result.get("total_discoveries", 0),
            "step_at_first_exploit": ep_result.get("step_at_first_exploit", -1),
        }
        episode_data.append(ep_info)

        # Phase 6.2: Auto-checkpoint every 5 episodes
        # NOTE: Guardrail checkpoint (below) handles per-agent PPO saves.
        # The auto-save here only fires if get_ppo_state_dicts() exists and
        # returns non-empty data; otherwise it's skipped to avoid creating
        # stale files that collide with the directory-based final save.
        if ckpt_mgr.should_auto_save(ep):
            try:
                state_dict = orch.get_ppo_state_dicts() if hasattr(orch, 'get_ppo_state_dicts') else {}
                if state_dict:  # Only save if we got real state data
                    ckpt_mgr.auto_save(state_dict, ep, metadata={
                        "episode": ep,
                        "avg_reward": sum(all_rewards[-10:]) / max(len(all_rewards[-10:]), 1),
                        "highest_phase": highest,
                    })
            except Exception as exc:
                console.print(f"[yellow]⚠️ Auto-checkpoint failed: {exc}[/yellow]")

        # Phase 6.6: Post-episode guardrail checkpoint
        ckpt_path = guardrails.post_episode_check(ep, difficulty_preset)
        if ckpt_path:
            try:
                orch.save_ppo_checkpoints(ckpt_path)
                guardrails.checkpoint.record_save(ckpt_path)
                if verbosity != "quiet":
                    console.print(f"[green]💾 Guardrail checkpoint saved: {ckpt_path}[/green]")
            except Exception as exc:
                logger.warning(f"Guardrail checkpoint failed: {exc}")

        recent = all_rewards[-10:]
        avg_recent = sum(recent) / len(recent)

        # Phase 6.9: Clean episode progress line — no Rich Progress overlay
        if verbosity != "quiet":
            ppo_str = ""
            if ppo_metrics.get("updates_fired", 0) > 0:
                ppo_str = (
                    f"PPO[upd:{ppo_metrics.get('updates_fired',0)} "
                    f"π:{ppo_metrics.get('avg_policy_loss',0):.3f} "
                    f"V:{ppo_metrics.get('avg_value_loss',0):.3f} "
                    f"H:{ppo_metrics.get('avg_entropy',0):.3f}]"
                )
            pct = (ep + 1) / episodes * 100
            elapsed_so_far = time.time() - start_time
            console.print(
                f"[bold cyan]━━ Ep {ep+1}/{episodes} ({pct:.0f}%) ━━[/bold cyan] "
                f"R:[bold {'green' if reward > 0 else 'red'}]{reward:+.0f}[/bold {'green' if reward > 0 else 'red'}]  "
                f"avg10:{avg_recent:+.0f}  "
                f"phase:[bold yellow]{highest}[/bold yellow]  "
                f"{ppo_str}  "
                f"[dim]{elapsed_so_far:.0f}s[/dim]"
            )

    elapsed = time.time() - start_time

    # Save PPO checkpoint
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    orch.save_ppo_checkpoints(checkpoint_path)

    # ── Summary ──────────────────────────────────────────────────────────
    avg_reward = sum(all_rewards) / len(all_rewards)
    max_reward = max(all_rewards)
    min_reward = min(all_rewards)
    last10_avg = sum(all_rewards[-10:]) / min(10, len(all_rewards))

    phase_counts = Counter(all_phases)
    phase_dist_str = "  ".join(f"{p}:{c}" for p, c in phase_counts.most_common())
    # CLOSEOUT implies EXFILTRATION was already reached — count both
    exfil_count = phase_counts.get("EXFILTRATION", 0) + phase_counts.get("CLOSEOUT", 0)
    exfil_pct = exfil_count / len(all_phases) * 100
    closeout_pct = phase_counts.get("CLOSEOUT", 0) / len(all_phases) * 100

    summary_table = Table(title="📊 ARIASKA Training Summary", box=box.ROUNDED)
    summary_table.add_column("Metric", style="cyan", width=24)
    summary_table.add_column("Value", style="bold white", width=30)
    summary_table.add_column("Quality", style="dim", width=16)
    summary_table.add_row("Episodes", str(episodes), "")
    summary_table.add_row("Steps/Episode", str(max_steps), "")
    summary_table.add_row("Duration", f"{elapsed:.1f}s ({elapsed/episodes:.2f}s/ep)", "")
    summary_table.add_row("Avg Reward", f"[bold green]{avg_reward:+.1f}[/bold green]",
                          f"{'🔥 excellent' if avg_reward > 1200 else '✅ good' if avg_reward > 800 else '⚠️ moderate'}")
    summary_table.add_row("Last-10 Avg", f"{last10_avg:+.1f}",
                          f"{'📈' if last10_avg > avg_reward else '📉'} vs overall")
    summary_table.add_row("Max Reward", f"[green]{max_reward:+.1f}[/green]", "")
    summary_table.add_row("Min Reward", f"{min_reward:+.1f}",
                          f"spread: {max_reward - min_reward:.0f}")
    summary_table.add_row("Reward StdDev", f"±{(sum((r - avg_reward)**2 for r in all_rewards) / max(len(all_rewards), 1))**0.5:.1f}",
                          f"{'🎯 consistent' if (sum((r - avg_reward)**2 for r in all_rewards) / max(len(all_rewards), 1))**0.5 < 200 else '📊 variable'}")
    summary_table.add_row("─" * 24, "─" * 24, "─" * 16)
    summary_table.add_row("📤 EXFILTRATION %", f"[bold {'green' if exfil_pct >= 80 else 'yellow'}]{exfil_pct:.0f}%[/bold {'green' if exfil_pct >= 80 else 'yellow'}] ({exfil_count}/{len(all_phases)} episodes)",
                          f"{'🏆 domination' if exfil_pct == 100 else '✅ strong' if exfil_pct >= 80 else '⚠️ needs work'}")
    summary_table.add_row("🧹 CLOSEOUT %", f"[bold {'green' if closeout_pct >= 30 else 'yellow'}]{closeout_pct:.0f}%[/bold {'green' if closeout_pct >= 30 else 'yellow'}] ({phase_counts.get('CLOSEOUT', 0)}/{len(all_phases)} episodes)",
                          f"{'🏆 clean' if closeout_pct == 100 else ''}")
    summary_table.add_row("─" * 24, "─" * 24, "─" * 16)
    summary_table.add_row("Phase Distribution", phase_dist_str, "")
    console.print(summary_table)

    # Decision source summary
    agg_sources: dict = {}
    for ed in episode_data:
        for k, v in ed.get("sources", {}).items():
            agg_sources[k] = agg_sources.get(k, 0) + v
    if agg_sources:
        total_decisions = sum(agg_sources.values()) or 1
        src_table = Table(title="🎯 Decision Sources", box=box.SIMPLE)
        src_table.add_column("Source", style="cyan")
        src_table.add_column("Count", style="white")
        src_table.add_column("Percent", style="yellow")
        for src, cnt in sorted(agg_sources.items(), key=lambda x: -x[1]):
            src_table.add_row(src, str(cnt), f"{cnt/total_decisions*100:.1f}%")
        console.print(src_table)

    # Save results
    results_path = f"artifacts/phase5_{episodes}ep_results.json"
    os.makedirs("artifacts", exist_ok=True)
    results = {
        "timestamp": datetime.now().isoformat(),
        "episodes": episodes,
        "max_steps": max_steps,
        "seed": seed,
        "target_ip": target_ip,
        "mode": mode,
        "duration_s": elapsed,
        "avg_reward": avg_reward,
        "last10_avg": last10_avg,
        "max_reward": max_reward,
        "min_reward": min_reward,
        "exfil_pct": exfil_pct,
        "closeout_pct": closeout_pct,
        "phase_distribution": dict(phase_counts),
        "decision_sources": agg_sources,
        "episode_data": episode_data,
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
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
        "🧠 Full RL training with PPO,\nplaybooks, 144 pentesting cmds",
        "ariaska smart-train --episodes 100",
    )
    help_table.add_row("  --episodes N", "Number of training episodes", "--episodes 50")
    help_table.add_row("  --steps N", "Max steps per episode (default 120)", "--steps 150")
    help_table.add_row("  --seed N", "Random seed (default 42)", "--seed 123")
    help_table.add_row("  --target IP", "Target IP for environment", "--target 192.168.56.101")
    help_table.add_row(
        "  --env ENV",
        "Environment preset:\nsim, msf, msf3, htb",
        "--env msf",
    )
    help_table.add_row("  --verbosity LEVEL", "quiet / standard / verbose", "--verbosity verbose")
    help_table.add_row("status", "Show system status & diagnostics", "ariaska status")
    help_table.add_row("help", "Show this help message", "ariaska help")

    console.print(Panel(
        help_table,
        title="🧠 ARIASKA_RL v5.0 — Cybersecurity RL Training",
        subtitle="Phase 5: Metasploitable 2 Ready",
        border_style="blue",
    ))


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
    train_p = sub.add_parser("smart-train", help="Run RL training")
    train_p.add_argument("--episodes", "-e", type=int, default=100, help="Number of episodes")
    train_p.add_argument("--steps", "-s", type=int, default=40, help="Max steps per episode (default: 40)")
    train_p.add_argument("--seed", type=int, default=42, help="Random seed")
    train_p.add_argument("--target", type=str, default=None, help="Target IP")
    train_p.add_argument("--env", type=str, default="ms3",
                         choices=["sim", "msf", "msf2", "ms2", "msf3", "ms3", "htb"],
                         help="Environment preset (default: msf = live MS3. Use ms2 for Metasploitable 2)")
    train_p.add_argument("--verbosity", "-v", type=str, default="verbose",
                         choices=["quiet", "standard", "verbose"])
    train_p.add_argument("--checkpoint", type=str, default=None,
                         help="PPO checkpoint path (auto-selected per mode if omitted)")
    
    # Phase 6.2: New flags
    train_p.add_argument("--dashboard", type=str, default="rich",
                         choices=["textual", "rich", "off"],
                         help="Dashboard mode (default: rich)")
    train_p.add_argument("--log-jsonl", action="store_true", default=True,
                         help="Enable JSONL event logging (default: on)")
    train_p.add_argument("--no-log-jsonl", action="store_true", default=False,
                         help="Disable JSONL event logging")
    train_p.add_argument("--mentor-budget", type=float, default=0.30,
                         help="Mentor call budget as fraction of steps (default: 0.30)")
    train_p.add_argument("--mentor-min-rate", type=float, default=0.15,
                         help="Minimum mentor call rate (default: 0.15)")
    train_p.add_argument("--mentor-max-rate", type=float, default=0.35,
                         help="Maximum mentor call rate (default: 0.35)")
    train_p.add_argument("--resume", type=str, default=None,
                         help="Resume from checkpoint path")
    
    # Phase 6.6: Difficulty presets & overnight guardrails
    train_p.add_argument("--difficulty", "-d", type=str, default="ms3_live",
                         choices=["normal","medium","hard","ms3_medium","ms3_hard","ms3_live"],
                         help="Difficulty preset: normal=all services, medium=no instant-root, hard=multi-step only, ms3_medium/ms3_hard/ms3_live=MS3-specific (default: ms3_live)")
    train_p.add_argument("--max-tokens-run", type=int, default=None,
                         help="Total token budget for the entire run (default: unlimited)")
    train_p.add_argument("--checkpoint-every", type=int, default=10,
                         help="Save PPO checkpoints every N episodes (default: 10)")

    # Phase 6.7: Anti-forensics & ethics mode
    train_p.add_argument("--anti-forensics", action="store_true", default=True,
                         help="Enable anti-forensics in CLOSEOUT phase (default: ON)")
    train_p.add_argument("--no-anti-forensics", dest="anti_forensics", action="store_false",
                         help="Disable anti-forensics in CLOSEOUT phase")
    train_p.add_argument("--ethics-mode", type=str, default="training",
                         choices=["training", "assessment", "demo"],
                         help="Ethics mode: training=full capability, assessment=audit logging, demo=investor-safe display (default: training)")
    train_p.add_argument("--seed-skills", action="store_true", default=True,
                         help="Pre-seed SkillLibrary with expert skill cards on startup (default: ON)")
    train_p.add_argument("--no-seed-skills", dest="seed_skills", action="store_false",
                         help="Disable pre-seeding of SkillLibrary")

    # Legacy positional support: smart-train <episodes> [env_flag]
    train_p.add_argument("pos_episodes", nargs="?", type=int, default=None)
    train_p.add_argument("pos_env", nargs="?", type=str, default=None)

    # Phase 7.1: ingest-htb removed — knowledge is now hardcoded in knowledge_packs.py

    sub.add_parser("status", help="Show system status")
    sub.add_parser("help", help="Show help")

    args = parser.parse_args()

    if args.command is None or args.command == "help":
        show_help()
        return

    if args.command == "status":
        show_system_status()
        return

    if args.command == "smart-train":
        ENV_PRESETS = {
            "sim": {"target_ip": "10.10.10.10", "mode": "simulated", "platform": "linux", "difficulty": "medium"},
            "msf": {"target_ip": os.environ.get("ARIASKA_MSF3_IP", "172.28.0.11"), "mode": "live", "platform": "linux", "difficulty": "medium"},
            "ms2": {"target_ip": os.environ.get("ARIASKA_MSF_IP", "172.28.0.10"), "mode": "live", "platform": "linux", "difficulty": "easy"},
            "msf2": {"target_ip": os.environ.get("ARIASKA_MSF_IP", "172.28.0.10"), "mode": "live", "platform": "linux", "difficulty": "easy"},
            "msf3": {"target_ip": os.environ.get("ARIASKA_MSF3_IP", "172.28.0.11"), "mode": "live", "platform": "linux", "difficulty": "medium"},
            "ms3": {"target_ip": os.environ.get("ARIASKA_MSF3_IP", "172.28.0.11"), "mode": "live", "platform": "linux", "difficulty": "medium"},
            "htb": {"target_ip": os.environ.get("ARIASKA_HTB_IP", "10.10.10.x"), "mode": "live", "platform": "unknown", "difficulty": "hard"},
        }

        # Per-mode checkpoint paths
        CHECKPOINT_PATHS = {
            "simulated": "models/enhanced/ppo_sim_checkpoint.pt",
            "live": "models/enhanced/ppo_live_checkpoint.pt",
        }

        episodes = args.pos_episodes or args.episodes
        env_key = args.env
        if args.pos_env:
            env_key = args.pos_env.lstrip("-")
            if env_key not in ENV_PRESETS:
                if "." in env_key:
                    ENV_PRESETS["custom"] = {"target_ip": env_key, "mode": "live", "platform": "unknown", "difficulty": "medium"}
                    env_key = "custom"
                else:
                    env_key = "ms3"  # Default to MS3 live

        preset = ENV_PRESETS.get(env_key, ENV_PRESETS["msf"])
        target_ip = args.target or preset["target_ip"]
        
        # Auto-select checkpoint path based on mode if not explicitly provided
        checkpoint_path = args.checkpoint or CHECKPOINT_PATHS.get(preset["mode"], "models/enhanced/ppo_checkpoint.pt")

        os.environ["ARIASKA_MODE"] = preset["mode"]
        if preset["mode"] == "live":
            os.environ["ARIASKA_LIVE_MODE"] = "true"
            os.environ["ARIASKA_TARGET_IP"] = target_ip

        try:
            results = run_training(
                episodes=episodes,
                max_steps=args.steps,
                seed=args.seed,
                target_ip=target_ip,
                mode=preset["mode"],
                platform=preset["platform"],
                difficulty=preset["difficulty"],
                verbosity=args.verbosity,
                checkpoint_path=checkpoint_path,
                dashboard_mode=args.dashboard,
                log_jsonl=args.log_jsonl and not args.no_log_jsonl,
                mentor_budget=args.mentor_budget,
                mentor_min_rate=args.mentor_min_rate,
                mentor_max_rate=args.mentor_max_rate,
                resume_path=args.resume,
                difficulty_preset=args.difficulty,
                max_tokens_run=args.max_tokens_run,
                checkpoint_every=args.checkpoint_every,
                anti_forensics=args.anti_forensics,
                ethics_mode=args.ethics_mode,
                seed_skills=args.seed_skills,
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
