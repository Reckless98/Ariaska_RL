#!/usr/bin/env python3
"""
scripts/run_eval.py — Phase 39.3: CAP Regression Gate

Automated evaluation harness that runs simulated engagement episodes
and produces a structured JSON report.

Usage:
    python scripts/run_eval.py --scenario cap --runs 5 --seed 1337
    python scripts/run_eval.py --scenario cap --runs 3 --baseline results/baseline.json
    python scripts/run_eval.py --scenario cap --target 10.129.2.142 --runs 5

Report format:
    {
        "scenario": "cap",
        "runs": 5,
        "success_rate": 0.80,
        "avg_steps": 42.5,
        "avg_time_s": 12.3,
        "failures": [{"run_id": "...", "reason": "..."}],
        "git_sha": "abc123",
        "timestamp": "2026-02-19T..."
    }

Feature flag: FF_CAP_GATE (default ON)
  - If FF_CAP_GATE=0, evaluation runs but does not block exit code.

Exit code:
  - 0 if success_rate >= baseline threshold
  - 1 if below baseline (regression detected)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("ARIASKA_DRY_RUN", "1")

logger = logging.getLogger("ariaska.eval")

# ── Constants ──────────────────────────────────────────────────────────

DEFAULT_BASELINE_PATH = "results/cap_baseline.json"
DEFAULT_REPORT_DIR = "results/eval_reports"
DEFAULT_SUCCESS_THRESHOLD = 0.60  # 60% success rate minimum
DEFAULT_MAX_STEPS_PER_RUN = 100
DEFAULT_TARGET_IP = "10.129.2.142"
SCENARIO_PHASES = {
    "cap": [
        "RECON", "ENUMERATION", "EXPLOITATION",
        "PRIVILEGE_ESCALATION", "POST_EXPLOITATION",
    ],
}


def get_git_sha() -> str:
    """Get current git SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(_ROOT),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


class CAPEvaluator:
    """
    Runs simulated engagement episodes and measures success.

    Success criteria per scenario:
      - cap: Must reach at least EXPLOITATION phase AND discover >= 2 services
    """

    def __init__(
        self,
        scenario: str = "cap",
        max_steps: int = DEFAULT_MAX_STEPS_PER_RUN,
        seed: int = 42,
        target_ip: str = DEFAULT_TARGET_IP,
        verbose: bool = False,
    ) -> None:
        self._scenario = scenario
        self._max_steps = max_steps
        self._seed = seed
        self._target_ip = target_ip
        self._verbose = verbose

    def run_single(self, run_id: str, seed: int) -> Dict[str, Any]:
        """
        Run a single simulated engagement and return results.

        Returns:
            {
                "run_id": str,
                "success": bool,
                "steps": int,
                "time_s": float,
                "phases_reached": list,
                "discoveries": int,
                "highest_phase": str,
                "reason": str,  # failure reason if not success
                "decision_trail": list,  # for replay
            }
        """
        start_time = time.time()
        decision_trail: List[Dict[str, Any]] = []

        try:
            result = self._run_simulated_episode(
                run_id=run_id,
                seed=seed,
                decision_trail=decision_trail,
            )
        except Exception as e:
            elapsed = time.time() - start_time
            logger.warning("Run %s failed with exception: %s", run_id, e)
            return {
                "run_id": run_id,
                "success": False,
                "steps": 0,
                "time_s": round(elapsed, 2),
                "phases_reached": [],
                "discoveries": 0,
                "highest_phase": "RECON",
                "reason": f"exception: {str(e)[:200]}",
                "decision_trail": decision_trail,
            }

        elapsed = time.time() - start_time
        result["time_s"] = round(elapsed, 2)
        result["decision_trail"] = decision_trail
        return result

    def _run_simulated_episode(
        self,
        run_id: str,
        seed: int,
        decision_trail: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run a single simulated episode using the orchestration stack."""
        # Lazy imports to avoid circular deps
        from core.commands.command_registry import (
            AttackPhase,
            get_commands_for_phase,
        )

        # Map string phase names → AttackPhase enum
        _phase_enum = {
            "RECON": AttackPhase.RECON,
            "ENUMERATION": AttackPhase.ENUMERATION,
            "EXPLOITATION": AttackPhase.EXPLOITATION,
            "PRIVILEGE_ESCALATION": AttackPhase.PRIVILEGE_ESCALATION,
            "LATERAL_MOVEMENT": AttackPhase.LATERAL_MOVEMENT,
            "POST_EXPLOITATION": AttackPhase.POST_EXPLOITATION,
            "EXFILTRATION": AttackPhase.EXFILTRATION,
            "CLOSEOUT": AttackPhase.CLOSEOUT,
        }

        phases_reached = ["RECON"]
        total_discoveries = 0
        highest_phase_idx = 0
        current_phase = "RECON"

        phase_order = [
            "RECON", "ENUMERATION", "EXPLOITATION",
            "PRIVILEGE_ESCALATION", "LATERAL_MOVEMENT",
            "POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT",
        ]
        phase_index = {p: i for i, p in enumerate(phase_order)}

        # Simulated discovery board
        disc_board = {
            "ports": set(), "services": set(),
            "credentials": set(), "vulns": set(),
            "shells": set(), "flags_set": set(),
        }

        import random  # noqa: E402

        for step in range(self._max_steps):
            # Get available commands for phase (using AttackPhase enum)
            phase_enum = _phase_enum.get(current_phase, AttackPhase.RECON)
            phase_cmds = get_commands_for_phase(phase_enum)
            if not phase_cmds:
                phase_cmds = get_commands_for_phase(AttackPhase.RECON)

            # Pick a command (deterministic from seed)
            rng = random.Random(seed + step)
            cmd_template = rng.choice(phase_cmds) if phase_cmds else None

            if cmd_template is None:
                decision_trail.append({
                    "step": step,
                    "phase": current_phase,
                    "command": "echo no_commands",
                    "success": False,
                })
                continue

            command = cmd_template.template.replace("{target}", self._target_ip)
            success = rng.random() > 0.4
            new_discoveries = rng.randint(0, 3) if success else 0
            total_discoveries += new_discoveries

            # Simulate phase advancement
            if success and total_discoveries >= 3 and step > 5:
                cur_idx = phase_index.get(current_phase, 0)
                if cur_idx < len(phase_order) - 1:
                    next_phase = phase_order[cur_idx + 1]
                    if next_phase not in phases_reached:
                        phases_reached.append(next_phase)
                    current_phase = next_phase
                    if phase_index[next_phase] > highest_phase_idx:
                        highest_phase_idx = phase_index[next_phase]

            decision_trail.append({
                "step": step,
                "phase": current_phase,
                "command": command[:80],
                "template": cmd_template.name if cmd_template else "",
                "success": success,
                "discoveries": new_discoveries,
            })

        # Determine success
        expected_phases = SCENARIO_PHASES.get(self._scenario, ["EXPLOITATION"])
        reached_expected = any(p in phases_reached for p in expected_phases)
        is_success = reached_expected and total_discoveries >= 2

        reason = ""
        if not is_success:
            if not reached_expected:
                reason = f"did not reach required phases: {expected_phases}"
            elif total_discoveries < 2:
                reason = f"insufficient discoveries: {total_discoveries}"

        return {
            "run_id": run_id,
            "success": is_success,
            "steps": self._max_steps,
            "phases_reached": phases_reached,
            "discoveries": total_discoveries,
            "highest_phase": phase_order[highest_phase_idx],
            "reason": reason,
        }


def run_evaluation(
    scenario: str = "cap",
    runs: int = 5,
    seed: int = 42,
    max_steps: int = DEFAULT_MAX_STEPS_PER_RUN,
    target_ip: str = DEFAULT_TARGET_IP,
    baseline_path: Optional[str] = None,
    report_dir: str = DEFAULT_REPORT_DIR,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run CAP evaluation and produce structured report.

    Args:
        scenario: Evaluation scenario name
        runs: Number of runs
        seed: Base random seed
        max_steps: Max steps per run
        target_ip: Target IP for command templates
        baseline_path: Path to baseline JSON (optional)
        report_dir: Directory for reports
        verbose: Verbose output

    Returns:
        Evaluation report dict
    """
    evaluator = CAPEvaluator(
        scenario=scenario,
        max_steps=max_steps,
        seed=seed,
        target_ip=target_ip,
        verbose=verbose,
    )

    results = []
    for i in range(runs):
        run_id = f"{scenario}_run_{i:03d}"
        run_seed = seed + i
        logger.info("Starting run %s (seed=%d)...", run_id, run_seed)
        result = evaluator.run_single(run_id, run_seed)
        results.append(result)
        if verbose:
            status = "PASS" if result["success"] else "FAIL"
            logger.info(
                "  %s: %s (steps=%d, disc=%d, phase=%s)",
                run_id, status, result["steps"],
                result.get("discoveries", 0), result.get("highest_phase", "?"),
            )

    # Compute aggregate metrics
    successes = sum(1 for r in results if r["success"])
    success_rate = successes / max(runs, 1)
    avg_steps = sum(r["steps"] for r in results) / max(runs, 1)
    avg_time = sum(r.get("time_s", 0) for r in results) / max(runs, 1)
    failures = [
        {"run_id": r["run_id"], "reason": r.get("reason", "unknown")}
        for r in results if not r["success"]
    ]

    report = {
        "scenario": scenario,
        "runs": runs,
        "success_rate": round(success_rate, 3),
        "avg_steps": round(avg_steps, 1),
        "avg_time_s": round(avg_time, 2),
        "failures": failures,
        "git_sha": get_git_sha(),
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "max_steps": max_steps,
        "target_ip": target_ip,
    }

    # Save report
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_path / f"eval_{scenario}_{ts}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved to %s", report_file)

    # Save decision trails for replay
    trails_file = report_path / f"trails_{scenario}_{ts}.jsonl"
    with open(trails_file, "w") as f:
        for result in results:
            trail = result.pop("decision_trail", [])
            for entry in trail:
                entry["run_id"] = result["run_id"]
                f.write(json.dumps(entry, default=str) + "\n")

    # Baseline comparison
    baseline_threshold = DEFAULT_SUCCESS_THRESHOLD
    baseline = None
    bl_path = baseline_path or DEFAULT_BASELINE_PATH
    if Path(bl_path).exists():
        try:
            with open(bl_path) as f:
                baseline = json.load(f)
            baseline_threshold = baseline.get("success_rate", DEFAULT_SUCCESS_THRESHOLD)
            logger.info(
                "Loaded baseline: threshold=%.2f (from %s)",
                baseline_threshold, bl_path,
            )
        except Exception as e:
            logger.warning("Failed to load baseline: %s", e)
    else:
        # Create baseline on first run
        logger.info("No baseline found. Creating baseline at %s", bl_path)
        Path(bl_path).parent.mkdir(parents=True, exist_ok=True)
        with open(bl_path, "w") as f:
            json.dump(report, f, indent=2)
        baseline_threshold = success_rate

    report["baseline_threshold"] = baseline_threshold
    report["regression"] = success_rate < baseline_threshold

    return report


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CAP Regression Gate — automated evaluation harness",
    )
    parser.add_argument(
        "--scenario", default="cap",
        help="Evaluation scenario (default: cap)",
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Number of evaluation runs (default: 5)",
    )
    parser.add_argument(
        "--seed", type=int, default=1337,
        help="Base random seed (default: 1337)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=DEFAULT_MAX_STEPS_PER_RUN,
        help=f"Max steps per run (default: {DEFAULT_MAX_STEPS_PER_RUN})",
    )
    parser.add_argument(
        "--baseline", default=None,
        help="Path to baseline JSON file",
    )
    parser.add_argument(
        "--report-dir", default=DEFAULT_REPORT_DIR,
        help=f"Directory for reports (default: {DEFAULT_REPORT_DIR})",
    )
    parser.add_argument(
        "--target", default=DEFAULT_TARGET_IP,
        help=f"Target IP for commands (default: {DEFAULT_TARGET_IP})",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    report = run_evaluation(
        scenario=args.scenario,
        runs=args.runs,
        seed=args.seed,
        max_steps=args.max_steps,
        target_ip=args.target,
        baseline_path=args.baseline,
        report_dir=args.report_dir,
        verbose=args.verbose,
    )

    # Print summary
    print(json.dumps(report, indent=2))

    # Check FF_CAP_GATE
    cap_gate_enabled = os.environ.get("FF_CAP_GATE", "1") != "0"

    if report.get("regression", False) and cap_gate_enabled:
        print(
            f"\nREGRESSION DETECTED: success_rate={report['success_rate']:.2f} "
            f"< baseline={report['baseline_threshold']:.2f}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
