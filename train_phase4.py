#!/usr/bin/env python3
"""
Phase 4 Training Runner — Direct SmartOrchestrator training
with per-role PPO, playbook curriculum, and discovery sharing.

Usage:
    python train_phase4.py [--episodes N] [--steps N] [--seed N]
"""

import os
import sys
import time
import random
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch

# Project root
sys.path.insert(0, str(Path(__file__).parent))

# Set runtime flags before any imports
os.environ.setdefault("ARIASKA_MODE", "simulation")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ariaska.phase4")


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(
    episodes: int = 10,
    max_steps: int = 50,
    target_ip: str = "10.10.10.10",
    seed: int = 42,
    verbosity: str = "standard",
) -> Dict[str, Any]:
    """
    Run Phase 4 training with SmartOrchestrator.

    Returns:
        Training results dict with per-episode metrics.
    """
    set_seed(seed)

    from core.runtime_flags import set_runtime_flags
    set_runtime_flags(offline=True, enable_llm=False, require_llm=False)

    from core.gpt_manager import GPTManager
    from core.environment.cyber_environment import CyberEnvironment
    from core.orchestration.smart_orchestrator import (
        SmartOrchestrator,
        SmartOrchestratorConfig,
    )

    # ── Initialize components ──
    env = CyberEnvironment(defer_reset=False)
    gpt = GPTManager(enable_llm=False, require_llm=False, offline=True)

    config = SmartOrchestratorConfig(
        model="gpt-4o-mini",
        mentor_mode="anneal",
        mentor_warmup_episodes=1,
        mentor_min_rate=0.05,
        mentor_max_rate=0.3,
        max_steps_per_episode=max_steps,
        default_target=target_ip,
        dashboard_enabled=True,
        dashboard_mode="live",
    )

    orch = SmartOrchestrator(
        env=env,
        gpt_manager=gpt,
        config=config,
        verbosity=verbosity,
    )

    logger.info(
        f"Phase 4 training: {episodes} episodes × {max_steps} steps, "
        f"seed={seed}, agents={list(orch.agents.keys())}"
    )

    # ── Training loop with rich reporting ──
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        console = Console()
        has_rich = True
    except ImportError:
        console = None
        has_rich = False

    all_results: List[Dict[str, Any]] = []
    rewards: List[float] = []
    phases_reached: List[str] = []
    best_reward = float("-inf")
    best_phase = "RECON"

    phase_order = [
        "RECON", "ENUMERATION", "EXPLOITATION",
        "PRIVILEGE_ESCALATION", "LATERAL_MOVEMENT",
        "POST_EXPLOITATION", "EXFILTRATION",
    ]

    start_time = time.time()

    # Load PPO checkpoints from previous run if available
    orch.load_ppo_checkpoints()

    for ep in range(episodes):
        ep_id = f"p4_ep{ep:04d}"

        metrics = orch.run_episode(
            episode_id=ep_id,
            episode_number=ep,
            max_steps=max_steps,
            target=target_ip,
        )

        all_results.append(metrics)
        ep_reward = metrics.get("total_reward", 0.0)
        rewards.append(ep_reward)

        # Highest phase reached
        progression = metrics.get("phase_progression", ["RECON"])
        highest = progression[-1] if progression else "RECON"
        phases_reached.append(highest)

        if ep_reward > best_reward:
            best_reward = ep_reward
            best_phase = highest

        # PPO metrics summary
        ppo_updates = metrics.get("ppo_updates_fired", 0)
        ppo_pl = metrics.get("ppo_avg_policy_loss", None)
        ppo_vl = metrics.get("ppo_avg_value_loss", None)
        ppo_ent = metrics.get("ppo_avg_entropy", None)
        
        # Decision source counts
        dec_ppo = metrics.get("decisions_ppo", 0)
        dec_playbook = metrics.get("decisions_playbook", 0)
        dec_registry = metrics.get("decisions_registry", 0)

        # Rolling avg
        window = min(10, len(rewards))
        avg_recent = sum(rewards[-window:]) / window

        # Phase index for comparison
        highest_idx = phase_order.index(highest) if highest in phase_order else 0

        # Print summary
        ppo_str = (
            f"upd={ppo_updates} π={ppo_pl:.4f} v={ppo_vl:.1f} H={ppo_ent:.3f}"
            if ppo_pl is not None else "warming up"
        )
        src_str = f"ppo={dec_ppo}/pb={dec_playbook}/reg={dec_registry}"

        if has_rich:
            color = "green" if ep_reward > 0 else "red" if ep_reward < -20 else "yellow"
            console.print(
                f"  [{color}]EP {ep:3d}/{episodes}[/{color}] | "
                f"R={ep_reward:+8.1f} Avg10={avg_recent:+8.1f} | "
                f"{highest:22s} ({highest_idx+1}/7) | "
                f"src: {src_str} | PPO: {ppo_str}"
            )
        else:
            print(
                f"EP {ep:3d}/{episodes} | "
                f"R={ep_reward:+.1f} Avg={avg_recent:+.1f} | "
                f"Phase={highest} ({highest_idx+1}/7) | "
                f"src: {src_str} | PPO: {ppo_str}"
            )

    total_time = time.time() - start_time

    # Save PPO checkpoints for next run
    orch.save_ppo_checkpoints()

    # ── Final Summary ──
    avg_reward = sum(rewards) / max(len(rewards), 1)
    avg_last10 = sum(rewards[-10:]) / min(len(rewards), 10)

    # Phase distribution
    phase_counts: Dict[str, int] = {}
    for p in phases_reached:
        phase_counts[p] = phase_counts.get(p, 0) + 1

    # Find best phase reached across all episodes
    all_phases_seen = set(phases_reached)
    overall_best_phase = max(
        all_phases_seen,
        key=lambda x: phase_order.index(x) if x in phase_order else 0,
        default="RECON",
    )

    summary = {
        "episodes": episodes,
        "total_time": total_time,
        "avg_reward": avg_reward,
        "avg_reward_last10": avg_last10,
        "best_reward": best_reward,
        "best_phase": best_phase,
        "overall_best_phase": overall_best_phase,
        "phase_distribution": phase_counts,
        "reward_history": rewards,
        "seed": seed,
    }

    if has_rich:
        table = Table(title="Phase 4 Training Summary", show_lines=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Episodes", str(episodes))
        table.add_row("Total Time", f"{total_time:.1f}s")
        table.add_row("Avg Reward", f"{avg_reward:+.2f}")
        table.add_row("Avg Reward (last 10)", f"{avg_last10:+.2f}")
        table.add_row("Best Episode Reward", f"{best_reward:+.2f}")
        table.add_row("Best Phase Reached", overall_best_phase)
        table.add_row("Phase Distribution", str(phase_counts))

        console.print()
        console.print(table)

        # Assessment
        if avg_last10 > 0:
            console.print(Panel(
                "✅ POSITIVE avg reward achieved! PPO is learning.",
                border_style="green",
            ))
        elif avg_last10 > -10:
            console.print(Panel(
                "⚠️ Near-zero rewards. PPO is exploring but not yet converging.",
                border_style="yellow",
            ))
        else:
            console.print(Panel(
                "❌ Negative rewards. May need more episodes or hyperparameter tuning.",
                border_style="red",
            ))
    else:
        print(f"\n=== Phase 4 Training Summary ===")
        print(f"Episodes: {episodes} | Time: {total_time:.1f}s")
        print(f"Avg Reward: {avg_reward:+.2f} | Last 10: {avg_last10:+.2f}")
        print(f"Best: {best_reward:+.2f} | Phase: {overall_best_phase}")
        print(f"Distribution: {phase_counts}")

    # Save results
    results_dir = Path("logs/phase4_training")
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    results_file = results_dir / f"phase4_run_{ts}.json"
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Training Runner")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--steps", type=int, default=50, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--target", type=str, default="10.10.10.10", help="Target IP")
    parser.add_argument("-v", "--verbosity", default="standard",
                        choices=["quiet", "standard", "verbose", "debug"])
    args = parser.parse_args()

    results = run_training(
        episodes=args.episodes,
        max_steps=args.steps,
        target_ip=args.target,
        seed=args.seed,
        verbosity=args.verbosity,
    )

    # Exit code based on performance
    if results["avg_reward_last10"] > 0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
