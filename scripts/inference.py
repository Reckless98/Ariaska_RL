"""Inference & Evaluation Baseline — Ariaska RL.

Runs PPO agent inference without training (eval-only mode) or evaluates
a checkpoint against baseline metrics for before/after comparison.

Usage:
    # Baseline evaluation (before training)
    python -m scripts.inference --model-path base --output results/baseline.json

    # Evaluate a specific checkpoint
    python -m scripts.inference --model-path models/distilled/h200_xxx_ep0100.pt --output results/after_training.json

    # Compare two results
    python -m scripts.inference --compare results/baseline.json results/after_training.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("ariaska.inference")

# ── Rich output ─────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
except ImportError:
    raise SystemExit("ERROR: 'rich' is required. Install with: pip install rich")

console = Console(force_terminal=True)

# ── Constants ────────────────────────────────────────────────────────────
STATE_DIM = 512
ACTION_DIM = 5
RESULTS_DIR = Path("results/eval")
PHASES = [
    "RECON", "ENUMERATION", "EXPLOITATION", "PRIVILEGE_ESCALATION",
    "LATERAL_MOVEMENT", "POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT",
]


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics for a checkpoint."""
    model_path: str = ""
    episodes: int = 0
    total_steps: int = 0
    avg_reward: float = 0.0
    median_reward: float = 0.0
    std_reward: float = 0.0
    avg_steps: float = 0.0
    max_phase_reached: str = "RECON"
    phase_distribution: Dict[str, int] = field(default_factory=dict)
    unique_commands: int = 0
    diversity_ratio: float = 0.0
    total_discoveries: int = 0
    avg_discoveries: float = 0.0
    step_at_first_exploit: Optional[int] = None
    completion_rate: float = 0.0
    reward_per_episode: List[float] = field(default_factory=list)
    wall_seconds: float = 0.0
    timestamp: str = ""


class InferenceRunner:
    """Runs eval-only episodes with a PPO checkpoint."""

    def __init__(
        self,
        model_path: str,
        episodes: int = 50,
        max_steps: int = 150,
        seed: int = 42,
        device: str = "auto",
    ):
        self.model_path = model_path
        self.episodes = episodes
        self.max_steps = max_steps
        self.seed = seed

        import torch
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._ppo: Any = None
        self._env: Any = None

    def _seed_all(self, seed: int) -> None:
        import torch
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _init(self) -> None:
        """Initialize PPO + env."""
        import torch
        from core.algorithms.ppo_agent import PPOAgent, PPOConfig

        self._seed_all(self.seed)

        config = PPOConfig(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
        )
        self._ppo = PPOAgent(config=config, device=str(self.device))

        # Load checkpoint
        if self.model_path and self.model_path != "base":
            if os.path.isfile(self.model_path):
                self._ppo.load(self.model_path)
                console.print(f"[green]✔ Loaded checkpoint:[/green] {self.model_path}")
            else:
                # Auto-detect best checkpoint
                loaded = self._auto_find_best()
                if loaded:
                    console.print(f"[green]✔ Auto-loaded:[/green] {loaded}")
                else:
                    console.print("[yellow]No checkpoint found — evaluating untrained model[/yellow]")
        else:
            console.print("[dim]Evaluating base (untrained) model[/dim]")

        from core.environment.cyber_environment import CyberEnvironment
        self._env = CyberEnvironment()

    def _auto_find_best(self) -> Optional[str]:
        """Find and load the best available checkpoint."""
        import re

        # Check distilled first
        distilled_dir = Path("models/distilled")
        if distilled_dir.is_dir():
            pattern = re.compile(r"h200_(.+?)_ep(\d+)\.pt$")
            candidates = []
            for f in distilled_dir.iterdir():
                m = pattern.match(f.name)
                if m:
                    candidates.append((f, m.group(1), int(m.group(2))))
            if candidates:
                latest_run = max(set(c[1] for c in candidates))
                best = max(
                    [c for c in candidates if c[1] == latest_run],
                    key=lambda c: c[2],
                )
                self._ppo.load(str(best[0]))
                return str(best[0])

        # Check enhanced
        enhanced_dir = Path("models/enhanced")
        if enhanced_dir.is_dir():
            for entry in sorted(enhanced_dir.iterdir(), reverse=True):
                if entry.is_dir():
                    for pt in entry.iterdir():
                        if pt.suffix == ".pt" and "ppo" in pt.name:
                            try:
                                self._ppo.load(str(pt))
                                return str(pt)
                            except Exception:
                                continue
        return None

    def run(self) -> EvalMetrics:
        """Run evaluation episodes and collect metrics."""
        self._init()

        metrics = EvalMetrics(
            model_path=self.model_path,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        all_rewards: List[float] = []
        all_steps: List[int] = []
        all_commands: set = set()
        total_discoveries = 0
        first_exploit_step: Optional[int] = None
        completions = 0
        phase_counts: Dict[str, int] = {}
        highest_phase_idx = 0

        wall_start = time.monotonic()

        console.print(Panel(
            f"Model: {self.model_path}\n"
            f"Episodes: {self.episodes}\n"
            f"Max steps: {self.max_steps}\n"
            f"Device: {self.device}\n"
            f"Seed: {self.seed}",
            title="Evaluation Config",
        ))

        for ep in range(self.episodes):
            state = self._env.reset()
            if isinstance(state, tuple):
                state = state[0] if isinstance(state[0], dict) else {"phase": "RECON"}
            if not isinstance(state, dict):
                state = self._env.get_global_state()

            ep_reward = 0.0
            ep_discoveries = 0
            max_phase = "RECON"

            for step_i in range(self.max_steps):
                from core.models.state_encoder import encode_state
                state_tensor = encode_state(
                    state, self.device,
                    current_step=step_i, max_steps=self.max_steps,
                )

                action_idx, _, _ = self._ppo.select_action(
                    state_tensor, training=False,
                )

                # Map action to command
                command = self._action_to_command(action_idx, state)
                all_commands.add(command)

                next_state, reward, done, info = self._env.step(command)
                if not isinstance(next_state, dict):
                    next_state = self._env.get_global_state()

                ep_reward += float(reward)

                # Track discoveries
                step_disc = info.get("discoveries", [])
                if isinstance(step_disc, list):
                    ep_discoveries += len(step_disc)

                # Track phase
                current_phase = next_state.get("phase", "RECON")
                if current_phase in PHASES:
                    pidx = PHASES.index(current_phase)
                    if pidx > PHASES.index(max_phase):
                        max_phase = current_phase
                    if pidx > highest_phase_idx:
                        highest_phase_idx = pidx
                    # First exploitation step
                    if current_phase == "EXPLOITATION" and first_exploit_step is None:
                        first_exploit_step = metrics.total_steps + step_i

                state = next_state
                if done:
                    break

            all_rewards.append(ep_reward)
            all_steps.append(step_i + 1)
            total_discoveries += ep_discoveries
            phase_counts[max_phase] = phase_counts.get(max_phase, 0) + 1
            metrics.total_steps += step_i + 1

            if max_phase in ("EXFILTRATION", "CLOSEOUT"):
                completions += 1

            if (ep + 1) % 10 == 0:
                avg = float(np.mean(all_rewards[-10:]))
                console.print(
                    f"  Ep {ep + 1}/{self.episodes} | "
                    f"avg10={avg:+.2f} | phase={max_phase}"
                )

        wall_elapsed = time.monotonic() - wall_start

        # Aggregate
        metrics.episodes = self.episodes
        metrics.avg_reward = float(np.mean(all_rewards))
        metrics.median_reward = float(np.median(all_rewards))
        metrics.std_reward = float(np.std(all_rewards))
        metrics.avg_steps = float(np.mean(all_steps))
        metrics.max_phase_reached = PHASES[highest_phase_idx]
        metrics.phase_distribution = phase_counts
        metrics.unique_commands = len(all_commands)
        metrics.diversity_ratio = len(all_commands) / max(metrics.total_steps, 1)
        metrics.total_discoveries = total_discoveries
        metrics.avg_discoveries = total_discoveries / max(self.episodes, 1)
        metrics.step_at_first_exploit = first_exploit_step
        metrics.completion_rate = completions / max(self.episodes, 1)
        metrics.reward_per_episode = all_rewards
        metrics.wall_seconds = wall_elapsed

        return metrics

    @staticmethod
    def _action_to_command(action_idx: int, state: Dict[str, Any]) -> str:
        """Map PPO action to command string."""
        target = state.get("target_ip", "172.28.128.3")
        try:
            from core.algorithms.command_action_mapper import CommandActionMapper
            role = state.get("role", "attacker")
            mapper = CommandActionMapper(role=role)
            tpl = mapper.action_to_command(action_idx)
            if tpl is not None:
                return tpl.template.replace("{target}", target)
        except Exception:
            pass
        # Fallback
        fallbacks = [
            f"nmap -sV -sC -T4 {target}",
            f"gobuster dir -u http://{target} -w /usr/share/wordlists/dirb/common.txt",
            f"searchsploit --nmap {target}",
            f"sudo -l",
            f"cat /etc/shadow",
        ]
        return fallbacks[action_idx % len(fallbacks)]


def print_metrics(m: EvalMetrics) -> None:
    """Print eval metrics as a Rich table."""
    table = Table(title="Evaluation Results", show_lines=True)
    table.add_column("Metric", style="cyan", min_width=25)
    table.add_column("Value", style="green", min_width=20)

    table.add_row("Model", str(m.model_path))
    table.add_row("Episodes", str(m.episodes))
    table.add_row("Total steps", str(m.total_steps))
    table.add_row("Wall time", f"{m.wall_seconds:.1f}s")
    table.add_row("─────────", "─────────")
    table.add_row("Avg reward", f"{m.avg_reward:+.3f}")
    table.add_row("Median reward", f"{m.median_reward:+.3f}")
    table.add_row("Std reward", f"{m.std_reward:.3f}")
    table.add_row("─────────", "─────────")
    table.add_row("Avg steps/episode", f"{m.avg_steps:.1f}")
    table.add_row("Max phase reached", m.max_phase_reached)
    table.add_row("Unique commands", str(m.unique_commands))
    table.add_row("Diversity ratio", f"{m.diversity_ratio:.4f}")
    table.add_row("Total discoveries", str(m.total_discoveries))
    table.add_row("Avg discoveries", f"{m.avg_discoveries:.2f}")
    table.add_row("First exploit step", str(m.step_at_first_exploit or "N/A"))
    table.add_row("Completion rate", f"{m.completion_rate:.1%}")

    for phase in PHASES:
        count = m.phase_distribution.get(phase, 0)
        if count > 0:
            table.add_row(f"  → {phase}", str(count))

    console.print(table)


def compare_results(before_path: str, after_path: str) -> None:
    """Compare two evaluation result files and print delta table."""
    with open(before_path) as f:
        before = json.load(f)
    with open(after_path) as f:
        after = json.load(f)

    table = Table(
        title="Before vs After Training Comparison",
        show_lines=True,
    )
    table.add_column("Metric", style="cyan", min_width=25)
    table.add_column("Before", style="yellow", min_width=15)
    table.add_column("After", style="green", min_width=15)
    table.add_column("Delta", min_width=15)

    compare_keys = [
        ("avg_reward", "Avg reward", "+.3f"),
        ("median_reward", "Median reward", "+.3f"),
        ("std_reward", "Std reward", ".3f"),
        ("avg_steps", "Avg steps", ".1f"),
        ("unique_commands", "Unique commands", "d"),
        ("diversity_ratio", "Diversity ratio", ".4f"),
        ("total_discoveries", "Total discoveries", "d"),
        ("avg_discoveries", "Avg discoveries", ".2f"),
        ("completion_rate", "Completion rate", ".1%"),
    ]

    for key, label, fmt in compare_keys:
        b = before.get(key, 0)
        a = after.get(key, 0)
        delta = a - b
        delta_str = f"{delta:{fmt}}"
        if isinstance(delta, (int, float)):
            if delta > 0:
                delta_str = f"[green]+{delta:{fmt}}[/green]"
            elif delta < 0:
                delta_str = f"[red]{delta:{fmt}}[/red]"
        table.add_row(label, f"{b:{fmt}}", f"{a:{fmt}}", delta_str)

    # Phase comparison
    b_phase = before.get("max_phase_reached", "RECON")
    a_phase = after.get("max_phase_reached", "RECON")
    b_idx = PHASES.index(b_phase) if b_phase in PHASES else 0
    a_idx = PHASES.index(a_phase) if a_phase in PHASES else 0
    phase_delta = a_idx - b_idx
    pd_str = f"+{phase_delta}" if phase_delta > 0 else str(phase_delta)
    if phase_delta > 0:
        pd_str = f"[green]{pd_str}[/green]"
    elif phase_delta < 0:
        pd_str = f"[red]{pd_str}[/red]"
    table.add_row("Max phase", b_phase, a_phase, pd_str)

    console.print(table)

    # Verdict
    improvement_score = 0
    if after.get("avg_reward", 0) > before.get("avg_reward", 0):
        improvement_score += 2
    if after.get("total_discoveries", 0) > before.get("total_discoveries", 0):
        improvement_score += 1
    if after.get("diversity_ratio", 0) > before.get("diversity_ratio", 0):
        improvement_score += 1
    if a_idx > b_idx:
        improvement_score += 2

    if improvement_score >= 4:
        console.print("\n[bold green]✔ VERDICT: Training was WORTH IT[/bold green]")
    elif improvement_score >= 2:
        console.print("\n[bold yellow]⚠ VERDICT: Marginal improvement — consider tuning[/bold yellow]")
    else:
        console.print("\n[bold red]✗ VERDICT: No improvement — check reward signals[/bold red]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ariaska RL — Inference & Evaluation Baseline",
    )
    parser.add_argument("--model-path", type=str, default="base",
                        help="Checkpoint path or 'base' for untrained (default: base)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path for results")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of eval episodes (default: 50)")
    parser.add_argument("--max-steps", type=int, default=150,
                        help="Max steps per episode (default: 150)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="torch device")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                        help="Compare two result JSON files")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Comparison mode
    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return

    # Evaluation mode
    runner = InferenceRunner(
        model_path=args.model_path,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        device=args.device,
    )

    metrics = runner.run()
    print_metrics(metrics)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.output or str(
        RESULTS_DIR / f"eval_{Path(args.model_path).stem}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out_path, "w") as f:
        json.dump(asdict(metrics), f, indent=2, default=str)
    console.print(f"\n[green]✔ Results saved:[/green] {out_path}")


if __name__ == "__main__":
    main()
