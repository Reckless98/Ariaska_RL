"""scripts/ablation_harness.py — C12: Systematic ablation harness.

Runs training episodes with each algorithm module disabled one at a
time, comparing against a baseline (all modules ON). Measures impact
via reward-invariant metrics (unique_commands, diversity_ratio,
total_discoveries, step_at_first_exploit).

Usage:
    python -m scripts.ablation_harness --episodes 5 --steps 50
    python -m scripts.ablation_harness --modules cognition_node sac_shadow
    python -m scripts.ablation_harness --dry-run

All modules are gated by feature flags (C07). The harness sets a flag
to False, runs episodes, resets, and moves to the next module.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("ariaska.ablation_harness")

# ─── Ablation module definitions ────────────────────────────────────

ABLATION_MODULES: Dict[str, str] = {
    "cognition_node": "CognitionNode multi-brain voting (C04)",
    "sac_shadow": "SAC shadow off-policy transitions (C03)",
    "reptile_meta": "Reptile meta-learning at episode boundaries (C08)",
    "source_win_rate_flag": "Per-source EMA win-rate tracking (C06)",
    "heldout_eval": "Held-out evaluation (C09, no direct impact on training)",
    "per_loss_grad_log": "Per-loss gradient norm logging (C05)",
    "use_micro_chain": "MicroChain 3-stage scoring (P27)",
    "evidence_graph": "Evidence graph reasoning",
    "hypothesis_engine": "Hypothesis engine",
    "bc_loss": "BC loss from TeacherTrace distillation (P14)",
    "teacher_trace": "Teacher trace collection for distillation",
    "neuromodulators": "Biologically-inspired neuromodulator control",
}

# Subset most commonly ablated (the algorithm modules from C01-C11)
DEFAULT_MODULES = [
    "cognition_node",
    "sac_shadow",
    "reptile_meta",
    "source_win_rate_flag",
    "use_micro_chain",
    "bc_loss",
    "neuromodulators",
]


@dataclass
class AblationMetrics:
    """Reward-invariant metrics for one ablation condition."""
    module: str  # "baseline" or flag name
    description: str
    episodes: int = 0
    total_steps: int = 0
    total_reward: float = 0.0
    unique_commands: int = 0
    unique_templates: int = 0
    total_discoveries: int = 0
    diversity_ratio: float = 0.0
    step_at_first_exploit: float = -1.0  # avg across episodes
    highest_phase: str = "RECON"
    wall_time_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module,
            "description": self.description,
            "episodes": self.episodes,
            "total_steps": self.total_steps,
            "total_reward": round(self.total_reward, 2),
            "unique_commands": self.unique_commands,
            "unique_templates": self.unique_templates,
            "total_discoveries": self.total_discoveries,
            "diversity_ratio": round(self.diversity_ratio, 4),
            "step_at_first_exploit": round(self.step_at_first_exploit, 1),
            "highest_phase": self.highest_phase,
            "wall_time_s": round(self.wall_time_s, 2),
        }


@dataclass
class AblationResult:
    """Complete ablation study result."""
    baseline: AblationMetrics
    ablations: Dict[str, AblationMetrics] = field(default_factory=dict)
    deltas: Dict[str, Dict[str, float]] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def compute_deltas(self) -> None:
        """Compute metric deltas relative to baseline."""
        b = self.baseline
        for name, m in self.ablations.items():
            self.deltas[name] = {
                "unique_commands": m.unique_commands - b.unique_commands,
                "unique_templates": m.unique_templates - b.unique_templates,
                "total_discoveries": m.total_discoveries - b.total_discoveries,
                "diversity_ratio": round(m.diversity_ratio - b.diversity_ratio, 4),
                "total_reward": round(m.total_reward - b.total_reward, 2),
            }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict(),
            "ablations": {k: v.to_dict() for k, v in self.ablations.items()},
            "deltas": self.deltas,
            "config": self.config,
        }


class AblationHarness:
    """Runs systematic ablation studies via feature flag toggling.

    For each module in the ablation set:
      1. Disable the module's feature flag
      2. Run N episodes with max_steps
      3. Collect reward-invariant metrics
      4. Re-enable the flag
      5. Compare against baseline

    Args:
        episodes_per_condition: How many episodes per ablation.
        max_steps_per_episode: Max steps per episode.
        target_ip: Target for CyberEnvironment.
        modules: Which modules to ablate. Defaults to DEFAULT_MODULES.
        output_dir: Where to save results JSON.
    """

    PHASE_RANK = {
        "RECON": 0, "ENUMERATION": 1, "EXPLOITATION": 2,
        "PRIVILEGE_ESCALATION": 3, "LATERAL_MOVEMENT": 4,
        "POST_EXPLOITATION": 5, "EXFILTRATION": 6, "CLOSEOUT": 7,
    }

    def __init__(
        self,
        episodes_per_condition: int = 3,
        max_steps_per_episode: int = 50,
        target_ip: str = "172.17.0.2",
        modules: Optional[Sequence[str]] = None,
        output_dir: str = "artifacts/ablation",
        seed: int = 42,
    ) -> None:
        self.episodes = episodes_per_condition
        self.max_steps = max_steps_per_episode
        self.target_ip = target_ip
        self.modules = list(modules or DEFAULT_MODULES)
        self.output_dir = Path(output_dir)
        self.seed = seed

        # Validate modules against known flags
        for m in self.modules:
            if m not in ABLATION_MODULES:
                logger.warning(f"Unknown ablation module: {m}")

    def _run_episodes(self, label: str) -> AblationMetrics:
        """Run N episodes and aggregate metrics.

        Uses SmartOrchestrator with simulated execution (dry run).
        """
        from core.environment.cyber_environment import CyberEnvironment
        from core.testing.fake_gpt_manager import FakeGPTManager
        from core.orchestration.smart_orchestrator import SmartOrchestrator

        agg = AblationMetrics(
            module=label,
            description=ABLATION_MODULES.get(label, "baseline (all ON)"),
        )

        t0 = time.monotonic()
        all_unique_cmds: set = set()
        all_unique_templates: set = set()
        exploit_steps: list = []
        best_phase = "RECON"

        for ep in range(self.episodes):
            env = CyberEnvironment(defer_reset=True)
            gpt = FakeGPTManager(seed=self.seed + ep)
            orch = SmartOrchestrator(
                env=env, gpt_manager=gpt, verbosity="silent",
            )

            try:
                # Run one training episode
                result = orch.run_training(
                    max_steps=self.max_steps,
                    target_ip=self.target_ip,
                )
            except Exception as e:
                logger.warning(f"[ABLATION] {label} ep={ep} error: {e}")
                result = {}

            # Extract metrics
            metrics = result if isinstance(result, dict) else {}
            agg.total_steps += metrics.get("total_steps", 0)
            agg.total_reward += metrics.get("total_reward", 0.0)
            agg.total_discoveries += metrics.get("total_discoveries", 0)

            for cmd in metrics.get("unique_commands_set", []):
                all_unique_cmds.add(cmd)
            all_unique_cmds.update(
                set(range(metrics.get("unique_commands_total", 0)))
            )

            for tmpl in metrics.get("unique_templates_set", []):
                all_unique_templates.add(tmpl)

            sfe = metrics.get("step_at_first_exploit", -1)
            if sfe >= 0:
                exploit_steps.append(sfe)

            hp = metrics.get("highest_phase", "RECON")
            if self.PHASE_RANK.get(hp, 0) > self.PHASE_RANK.get(best_phase, 0):
                best_phase = hp

        agg.episodes = self.episodes
        agg.unique_commands = metrics.get("unique_commands_total", len(all_unique_cmds))
        agg.unique_templates = metrics.get("unique_templates_total", len(all_unique_templates))
        agg.diversity_ratio = (
            agg.unique_commands / max(agg.total_steps, 1)
        )
        agg.step_at_first_exploit = (
            sum(exploit_steps) / len(exploit_steps) if exploit_steps else -1.0
        )
        agg.highest_phase = best_phase
        agg.wall_time_s = time.monotonic() - t0

        return agg

    def run(self) -> AblationResult:
        """Execute the full ablation study.

        Returns:
            AblationResult with baseline, per-module metrics, and deltas.
        """
        from core.feature_flags import (
            get_feature_flags,
            reset_feature_flags,
            set_feature_flag,
        )

        result = AblationResult(
            baseline=AblationMetrics(module="baseline", description="all ON"),
            config={
                "episodes_per_condition": self.episodes,
                "max_steps_per_episode": self.max_steps,
                "target_ip": self.target_ip,
                "modules": self.modules,
                "seed": self.seed,
            },
        )

        # 1. Baseline: all modules at default
        logger.info("[ABLATION] Running baseline...")
        reset_feature_flags()
        result.baseline = self._run_episodes("baseline")
        logger.info(
            f"[ABLATION] Baseline done: {result.baseline.total_discoveries} "
            f"discoveries, {result.baseline.unique_commands} unique cmds"
        )

        # 2. Each ablation
        for module in self.modules:
            logger.info(f"[ABLATION] Ablating: {module}...")
            reset_feature_flags()

            # Disable this module
            try:
                set_feature_flag(module, False)
            except ValueError:
                logger.warning(f"[ABLATION] Flag {module} not found, skipping.")
                continue

            m = self._run_episodes(module)
            result.ablations[module] = m

            logger.info(
                f"[ABLATION] {module} done: {m.total_discoveries} disc, "
                f"{m.unique_commands} cmds (Δ={m.unique_commands - result.baseline.unique_commands})"
            )

            # Re-enable
            reset_feature_flags()

        # 3. Compute deltas
        result.compute_deltas()

        # 4. Save
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / f"ablation_{int(time.time())}.json"
        out_path.write_text(json.dumps(result.to_dict(), indent=2))
        logger.info(f"[ABLATION] Results saved to {out_path}")

        # 5. Final reset
        reset_feature_flags()

        return result

    def print_report(self, result: AblationResult) -> None:
        """Print a Rich table summarizing the ablation study."""
        try:
            from rich.console import Console
            from rich.table import Table
        except ImportError:
            # Fallback: plain text
            print(json.dumps(result.to_dict(), indent=2))
            return

        console = Console()
        table = Table(
            title="Ablation Study Results",
            show_lines=True,
            title_style="bold cyan",
        )
        table.add_column("Module", style="bold")
        table.add_column("Discoveries", justify="right")
        table.add_column("Δ Disc", justify="right")
        table.add_column("Uniq Cmds", justify="right")
        table.add_column("Diversity", justify="right")
        table.add_column("Reward", justify="right")
        table.add_column("Phase", justify="center")
        table.add_column("Time (s)", justify="right")

        # Baseline row
        b = result.baseline
        table.add_row(
            "[green]BASELINE (all ON)[/green]",
            str(b.total_discoveries),
            "—",
            str(b.unique_commands),
            f"{b.diversity_ratio:.3f}",
            f"{b.total_reward:.1f}",
            b.highest_phase,
            f"{b.wall_time_s:.1f}",
        )

        # Ablation rows
        for name, m in result.ablations.items():
            d = result.deltas.get(name, {})
            dd = d.get("total_discoveries", 0)
            color = "red" if dd < 0 else "green" if dd > 0 else "white"
            table.add_row(
                f"[yellow]−{name}[/yellow]",
                str(m.total_discoveries),
                f"[{color}]{dd:+d}[/{color}]",
                str(m.unique_commands),
                f"{m.diversity_ratio:.3f}",
                f"{m.total_reward:.1f}",
                m.highest_phase,
                f"{m.wall_time_s:.1f}",
            )

        console.print(table)


def main() -> None:
    """CLI entry point for ablation harness."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ariaska ablation harness — systematic module evaluation"
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="Episodes per condition (default: 3)",
    )
    parser.add_argument(
        "--steps", type=int, default=50,
        help="Max steps per episode (default: 50)",
    )
    parser.add_argument(
        "--modules", nargs="*", default=None,
        help="Specific modules to ablate (default: all)",
    )
    parser.add_argument(
        "--target", type=str, default="172.17.0.2",
        help="Target IP (default: 172.17.0.2)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed (default: 42)",
    )
    parser.add_argument(
        "--output", type=str, default="artifacts/ablation",
        help="Output directory (default: artifacts/ablation)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print config and exit without running",
    )
    parser.add_argument(
        "--list-modules", action="store_true",
        help="List all ablatable modules and exit",
    )
    args = parser.parse_args()

    if args.list_modules:
        for name, desc in ABLATION_MODULES.items():
            marker = "★" if name in DEFAULT_MODULES else " "
            print(f"  {marker} {name:28s}  {desc}")
        return

    harness = AblationHarness(
        episodes_per_condition=args.episodes,
        max_steps_per_episode=args.steps,
        target_ip=args.target,
        modules=args.modules,
        output_dir=args.output,
        seed=args.seed,
    )

    if args.dry_run:
        from rich.console import Console
        Console().print(f"[bold]Ablation config:[/bold] {harness.episodes} ep × "
                        f"{harness.max_steps} steps × {len(harness.modules)} modules "
                        f"= {harness.episodes * (1 + len(harness.modules))} total runs")
        Console().print(f"Modules: {', '.join(harness.modules)}")
        return

    os.environ["ARIASKA_DRY_RUN"] = "1"
    result = harness.run()
    harness.print_report(result)


if __name__ == "__main__":
    main()
